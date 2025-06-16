import asyncio
import logging
import json
import re
import shutil
import random
import string
import hashlib
import aiosqlite
import signal
import sys
from datetime import datetime, timezone
from collections import deque
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from telethon import TelegramClient, events, errors
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument, MessageEntity
from PIL import Image
import io
import imagehash
import pyahocorasick
import os
from dotenv import load_dotenv
from difflib import SequenceMatcher

# Configuration
load_dotenv()
API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')
OWNER_ID_ENV = os.getenv('OWNER_ID')
if not API_ID or not API_HASH:
    raise ValueError("API_ID and API_HASH must be set in environment variables.")
API_ID = int(API_ID)
OWNER_ID = int(OWNER_ID_ENV) if OWNER_ID_ENV else None  # Initialize from .env
SESSION_FILE = "ghostcopy.session"
MAPPINGS_FILE = "channel_mappings.json"
TEMP_DIR = Path("./temp_images")
OUTPUT_DIR = Path("./output_images")
CACHE_DB_PATH = "./image_cache.db"
MAX_RETRIES = 3
RETRY_DELAY = 5
MAX_QUEUE_SIZE = 100
MAX_MAPPING_HISTORY = 1000
MAX_MESSAGE_LENGTH = 4096
NUM_WORKERS = 5
DEFAULT_DELAY_RANGE = [3.0, 7.0]
DEFAULT_DELAY_OFFSET = 1.0
MAX_CHATS_PER_SESSION = 70

# Initialize directories
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("ghostcopybotpro.log"), logging.StreamHandler()]
)
logger = logging.getLogger("GhostCopyBotPro")

# Initialize single client
client = TelegramClient(SESSION_FILE, API_ID, API_HASH)
client.forwarded_messages = {}  # type: Dict[str, int]

# Initialize async SQLite connection
db_connection = None

async def init_cache_db():
    """Initialize SQLite database for image cache."""
    async with aiosqlite.connect(CACHE_DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS image_cache (
                image_hash TEXT PRIMARY KEY,
                timestamp TEXT
            )
        """)
        await db.commit()

async def is_image_cached(image_hash: str) -> bool:
    """Check if image hash exists in cache."""
    async with aiosqlite.connect(CACHE_DB_PATH) as db:
        async with db.execute("SELECT 1 FROM image_cache WHERE image_hash = ?", (image_hash,)) as cursor:
            return await cursor.fetchone() is not None

async def cache_image(image_hash: str):
    """Store image hash in cache with timestamp."""
    async with aiosqlite.connect(CACHE_DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO image_cache (image_hash, timestamp) VALUES (?, ?)",
            (image_hash, datetime.utcnow().isoformat())
        )
        await db.commit()

# Data structures
channel_mappings = {}  # type: Dict[str, Dict[str, Dict[str, Any]]]
message_queue = deque(maxlen=MAX_QUEUE_SIZE)  # type: deque
queue_semaphore = asyncio.Semaphore(NUM_WORKERS)
is_connected = False
is_processing_enabled = True
pair_stats = {}  # type: Dict[str, Dict[str, Dict[str, Any]]]
shutdown_event = asyncio.Event()

def calculate_image_hash(file_data: bytes) -> str:
    """Calculate SHA256 hash of image data."""
    sha256 = hashlib.sha256()
    sha256.update(file_data)
    return sha256.hexdigest()

def save_mappings():
    """Save channel mappings to file."""
    try:
        with open(MAPPINGS_FILE, "w") as f:
            json.dump(channel_mappings, f, indent=2)
        logger.info("Channel mappings saved successfully.")
    except Exception as e:
        logger.error(f"Error saving mappings: {e}")

def load_mappings():
    """Load channel mappings from file."""
    global channel_mappings
    try:
        if os.path.exists(MAPPINGS_FILE):
            with open(MAPPINGS_FILE, "r") as f:
                channel_mappings = json.load(f)
            logger.info(f"Loaded {sum(len(v) for v in channel_mappings.values())} mappings.")
            for user_id, pairs in channel_mappings.items():
                if user_id not in pair_stats:
                    pair_stats[user_id] = {}
                for pair_name, mapping in pairs.items():
                    mapping.setdefault('header_patterns', [])
                    mapping.setdefault('footer_patterns', [])
                    mapping.setdefault('remove_phrases', [])
                    mapping.setdefault('remove_mentions', False)
                    mapping.setdefault('trap_phrases', [])
                    mapping.setdefault('honeypot_phrases', ['do_not_copy', 'trapword'])
                    mapping.setdefault('trap_image_hashes', [])
                    mapping.setdefault('delay_range', DEFAULT_DELAY_RANGE)
                    mapping.setdefault('delay_offset', DEFAULT_DELAY_OFFSET)
                    mapping.setdefault('status', 'active')
                    mapping.setdefault('last_activity', None)
                    mapping.setdefault('custom_header', '')
                    mapping.setdefault('custom_footer', '')
                    pair_stats[user_id][pair_name] = {
                        'forwarded': 0, 'edited': 0, 'deleted': 0, 'blocked': 0, 'queued': 0, 'last_activity': None
                    }
        else:
            logger.info("No mappings file found. Starting fresh.")
    except json.JSONDecodeError as e:
        logger.error(f"Corrupted mappings file: {e}. Backing up.")
        shutil.move(MAPPINGS_FILE, MAPPINGS_FILE + ".bak")
        channel_mappings = {}
    except Exception as e:
        logger.error(f"Error loading mappings: {e}")

def compile_patterns(patterns: List[str]) -> Optional[re.Pattern]:
    """Compile regex patterns for filtering with word boundaries."""
    if not patterns:
        return None
    escaped = [r'\b' + re.escape(p.strip().lower()) + r'\b' for p in patterns if p.strip()]
    return re.compile('|'.join(escaped), re.IGNORECASE) if escaped else None

def normalize_text(text: str) -> str:
    """Normalize text by removing invisible characters and extra spaces."""
    if not text:
        return ""
    text = re.sub(r'[\u200B-\u200F\uFEFF]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_patterns(text: str, patterns: List[str]) -> str:
    """Remove lines matching patterns from text."""
    if not text or not patterns:
        return text
    compiled = compile_patterns(patterns)
    if compiled:
        lines = text.split('\n')
        filtered_lines = [line for line in lines if not compiled.match(normalize_text(line).strip().lower())]
        return '\n'.join(filtered_lines).strip()
    return text

def remove_phrases(text: str, phrases: List[str]) -> Tuple[str, bool]:
    """Remove specified phrases from text."""
    if not text or not phrases:
        return text, False
    normalized_text = normalize_text(text)
    automaton = pyahocorasick.Automaton()
    for idx, phrase in enumerate(phrases):
        automaton.add_word(normalize_text(phrase).lower(), (idx, phrase))
    automaton.make_automaton()
    found = False
    result = []
    last_end = 0
    for end, (idx, phrase) in automaton.iter(normalized_text.lower()):
        result.append(text[last_end:end - len(phrase)])
        last_end = end
        found = True
    result.append(text[last_end:])
    cleaned_text = ''.join(result)
    return re.sub(r'\s+', ' ', cleaned_text).strip(), found

def paraphrase_phrases(text: str, phrases: List[str]) -> str:
    """Paraphrase specified phrases with synonyms."""
    if not text or not phrases:
        return text
    synonym_map = {
        'buy': ['purchase', 'acquire'],
        'sell': ['vend', 'trade'],
        'signal': ['indication', 'cue'],
        'profit': ['gain', 'return']
    }
    normalized_text = normalize_text(text)
    for phrase in phrases:
        normalized_phrase = normalize_text(phrase).lower()
        if normalized_phrase in synonym_map:
            synonyms = synonym_map[normalized_phrase]
            text = re.sub(
                r'\b' + re.escape(phrase) + r'\b',
                random.choice(synonyms),
                text,
                flags=re.IGNORECASE
            )
    return text

def clean_image_exif(image: Image.Image) -> Image.Image:
    """Remove EXIF metadata from image."""
    try:
        image = image.convert('RGB')
        new_image = Image.new('RGB', image.size)
        new_image.putdata(image.getdata())
        return new_image
    except Exception as e:
        logger.error(f"Error cleaning EXIF: {e}")
        return image

def obfuscate_image(image: Image.Image) -> Image.Image:
    """Apply subtle transformations to obfuscate image."""
    try:
        original_size = image.size
        scale = random.uniform(0.98, 1.02)
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        image = image.resize(new_size, Image.LANCZOS)
        image = image.resize(original_size, Image.LANCZOS)
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=random.randint(90, 95))
        return Image.open(output)
    except Exception as e:
        logger.error(f"Error obfuscating image: {e}")
        return image

async def process_image(event: events.NewMessage.Event, mapping: Dict[str, Any]) -> Tuple[Optional[io.BytesIO], bool]:
    """Process image for trap detection and stealth reupload."""
    input_path = None
    try:
        photo = await client.download_media(event.message, bytes)
        if not photo:
            logger.warning("Failed to download image.")
            return None, False

        image_hash = calculate_image_hash(photo)
        if await is_image_cached(image_hash):
            logger.info("Skipping cached image")
            random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            file_ext = 'jpg'
            file_bytes = io.BytesIO(photo)
            file_bytes.name = f"img_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{random_string}.{file_ext}"
            return file_bytes, False

        image = Image.open(io.BytesIO(photo))
        original_format = image.format or 'JPEG'
        original_size = image.size
        image = clean_image_exif(image)
        image = obfuscate_image(image)

        phash = str(imagehash.phash(image))
        if phash in mapping.get('trap_image_hashes', []):
            reason = f"Block image hash match: {phash}"
            await notify_trap(event, mapping, mapping['pair_name'], reason)
            return None, True

        random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        output = io.BytesIO()
        image.save(output, format=original_format, quality=random.randint(90, 95))
        file_ext = original_format.lower()
        file_bytes = io.BytesIO(output.getvalue())
        file_bytes.name = f"img_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{random_string}.{file_ext}"

        await cache_image(image_hash)
        return file_bytes, False
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None, False
    finally:
        if input_path and input_path.exists():
            for _ in range(3):
                try:
                    input_path.unlink()
                    logger.debug(f"Deleted temporary file: {input_path}")
                    break
                except Exception as e:
                    logger.error(f"Failed to delete {input_path}: {e}")
                    await asyncio.sleep(0.1)

async def notify_trap(event: events.NewMessage.Event, mapping: Dict[str, Any], pair_name: str, reason: str):
    """Notify owner of trap detection."""
    global OWNER_ID
    if not OWNER_ID:
        logger.warning(f"Cannot notify trap: OWNER_ID not set. Reason: {reason} in pair '{pair_name}'")
        return
    msg_id = getattr(event.message, 'id', 'Unknown')
    try:
        await client.send_message(
            OWNER_ID,
            f"🚫 Block detected in pair '{pair_name}' from '{mapping['source']}'.\n"
            f"🚫 Reason: {reason}\n🚫 Source Message ID: {msg_id}"
        )
        logger.info(f"Block hit: {reason} in pair '{pair_name}'")
    except Exception as e:
        logger.error(f"Failed to notify trap: {e}")

async def send_split_message(
    client: TelegramClient,
    entity: int,
    message_text: str,
    reply_to: Optional[int] = None,
    silent: bool = False,
    entities: Optional[List] = None
) -> Optional[Any]:
    """Send long messages in parts with rate limit handling."""
    parts = [message_text[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(message_text), MAX_MESSAGE_LENGTH)]
    sent_messages = []
    for attempt in range(MAX_RETRIES):
        try:
            for part in parts:
                sent_msg = await client.send_message(
                    entity=entity,
                    message=part,
                    reply_to=reply_to if not sent_messages else None,
                    silent=silent,
                    parse_mode='html',
                    formatting_entities=entities if entities and not sent_messages else None
                )
                sent_messages.append(sent_msg)
                await asyncio.sleep(random.uniform(0.1, 0.5))
            return sent_messages[0] if sent_messages else None
        except errors.FloodWaitError as e:
            wait_time = e.seconds + random.uniform(0, 1)
            logger.warning(f"Flood wait error, sleeping for {wait_time}s")
            await asyncio.sleep(wait_time)
        except errors.PeerFloodError:
            logger.error("Peer flood error, stopping message send")
            return None
        except Exception as e:
            logger.error(f"Error sending split message: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                raise
    return None

def adjust_entities(original_text: str, processed_text: str, entities: List[MessageEntity]) -> List[MessageEntity]:
    """Adjust message entities to align with modified text."""
    if not entities or not original_text or not processed_text:
        return None
    matcher = SequenceMatcher(None, original_text, processed_text)
    new_entities = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for entity in entities:
                if entity.offset >= i1 and entity.offset + entity.length <= i2:
                    new_offset = j1 + (entity.offset - i1)
                    new_entities.append(entity.__class__(offset=new_offset, length=entity.length, **{
                        k: v for k, v in entity.__dict__.items() if k not in ['offset', 'length']
                    }))
    return new_entities if new_entities else None

async def copy_message_with_retry(
    event: events.NewMessage.Event,
    mapping: Dict[str, Any],
    user_id: str,
    pair_name: str
) -> bool:
    """Copy message with retry logic."""
    source_msg_id = event.message.id if hasattr(event.message, 'id') else "Unknown"
    async with queue_semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                if not is_processing_enabled:
                    logger.info(f"Processing paused for pair '{pair_name}'")
                    await asyncio.sleep(1)
                    continue
                message_text = event.message.raw_text or ""
                normalized_text = normalize_text(message_text)
                text_lower = normalized_text.lower()
                original_entities = event.message.entities or []
                media = event.message.media
                reply_to = await handle_reply_mapping(event, mapping)

                compiled_honeypots = compile_patterns(mapping.get('honeypot_phrases', []))
                if compiled_honeypots and compiled_honeypots.search(text_lower):
                    logger.info(f"Skipped message due to honeypot phrase in pair '{pair_name}'")
                    return True

                compiled_traps = compile_patterns(mapping.get('trap_phrases', []))
                if compiled_traps and compiled_traps.search(text_lower):
                    reason = "Block phrase in text"
                    await notify_trap(event, mapping, pair_name, reason)
                    pair_stats[user_id][pair_name]['blocked'] += 1
                    return True

                processed_text = message_text
                processed_entities = original_entities
                if message_text:
                    processed_text = normalized_text
                    processed_text = remove_patterns(processed_text, mapping.get('header_patterns', []))
                    processed_text = remove_patterns(processed_text, mapping.get('footer_patterns', []))
                    processed_text, phrases_removed = remove_phrases(processed_text, mapping.get('remove_phrases', []))
                    if phrases_removed:
                        processed_text = paraphrase_phrases(processed_text, mapping.get('remove_phrases', []))
                    if mapping.get('remove_mentions', False):
                        processed_text = re.sub(r'@[a-zA-Z0-9_]+|t\.me/[^\s]+', '', processed_text)
                        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
                    processed_text = apply_custom_header_footer(
                        processed_text, mapping.get('custom_header', ''), mapping.get('custom_footer', '')
                    )
                    if normalize_text(processed_text).strip().lower() != normalize_text(message_text).strip().lower():
                        processed_entities = adjust_entities(message_text, processed_text, original_entities)
                        logger.info("Text changed, adjusted entities.")

                processed_media = None
                if isinstance(media, MessageMediaPhoto):
                    processed_media, is_trapped = await process_image(event, mapping)
                    if is_trapped:
                        pair_stats[user_id][pair_name]['blocked'] += 1
                        return True
                    if not processed_media:
                        logger.warning(f"Failed to process image for pair '{pair_name}'. Falling back to original.")
                        processed_media = media
                elif isinstance(media, MessageMediaDocument):
                    processed_media = media

                min_delay, max_delay = mapping.get('delay_range', DEFAULT_DELAY_RANGE)
                delay_offset = mapping.get('delay_offset', DEFAULT_DELAY_OFFSET)
                dest_jitter = hash(mapping['destination']) % 4 + random.uniform(1, 2)
                total_delay = random.uniform(min_delay, max_delay) + dest_jitter
                await asyncio.sleep(max(total_delay, 0))

                if processed_media or processed_text.strip():
                    sent_message = await client.send_message(
                        entity=int(mapping['destination']),
                        file=processed_media,
                        message=processed_text,
                        reply_to=reply_to,
                        silent=event.message.silent,
                        parse_mode='html',
                        formatting_entities=processed_entities
                    )
                    await store_message_mapping(event, mapping, sent_message)
                    pair_stats[user_id][pair_name]['forwarded'] += 1
                    pair_stats[user_id][pair_name]['last_activity'] = datetime.now().isoformat()
                    logger.info(f"Copied message from {mapping['source']} to {mapping['destination']} (ID: {sent_message.id})")
                    return True
                else:
                    reason = "Empty message after filtering"
                    await notify_trap(event, mapping, pair_name, reason)
                    pair_stats[user_id][pair_name]['blocked'] += 1
                    return True

            except errors.FloodWaitError as e:
                wait_time = e.seconds + random.uniform(0, 1)
                logger.warning(f"Flood wait error, sleeping for {wait_time}s for pair '{pair_name}' (Msg ID: {source_msg_id})")
                await asyncio.sleep(wait_time)
            except errors.ChatWriteForbiddenError:
                logger.warning(f"Bot forbidden to write in {mapping['destination']}. Pausing pair '{pair_name}'.")
                mapping['status'] = 'paused'
                save_mappings()
                if OWNER_ID:
                    await client.send_message(OWNER_ID, f"⏸️ Paused pair '{pair_name}' due to write permission error.")
                return False
            except errors.ChannelInvalidError:
                logger.warning(f"Invalid channel {mapping['destination']}. Pausing pair '{pair_name}'.")
                mapping['status'] = 'paused'
                save_mappings()
                if OWNER_ID:
                    await client.send_message(OWNER_ID, f"⏸️ Paused pair '{pair_name}' due to invalid channel.")
                return False
            except errors.MessageIdInvalidError:
                logger.warning(f"Invalid message ID for pair '{pair_name}'. Skipping.")
                return False
            except errors.PeerFloodError:
                logger.error(f"Peer flood error for pair '{pair_name}'. Pausing pair.")
                mapping['status'] = 'paused'
                save_mappings()
                if OWNER_ID:
                    await client.send_message(OWNER_ID, f"⏸️ Paused pair '{pair_name}' due to peer flood error.")
                return False
            except Exception as e:
                logger.error(f"Error copying message for pair '{pair_name}' (Msg ID: {source_msg_id}): {e}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                else:
                    if OWNER_ID:
                        await client.send_message(OWNER_ID, f"❌ Failed to copy message for pair '{pair_name}' after {MAX_RETRIES} attempts.")
                    return False

async def edit_copied_message(
    event: events.MessageEdited.Event,
    mapping: Dict[str, Any],
    user_id: str,
    pair_name: str
):
    """Edit copied message in destination channel."""
    try:
        mapping_key = f"{mapping['source']}:{event.message.id}"
        if mapping_key not in client.forwarded_messages:
            return

        forwarded_msg_id = client.forwarded_messages[mapping_key]
        message_text = event.message.raw_text or ""
        normalized_text = normalize_text(message_text)
        text_lower = normalized_text.lower()
        original_entities = event.message.entities or []
        media = event.message.media
        reply_to = await handle_reply_mapping(event, mapping)

        compiled_honeypots = compile_patterns(mapping.get('honeypot_phrases', []))
        if compiled_honeypots and compiled_honeypots.search(text_lower):
            await client.delete_messages(int(mapping['destination']), [forwarded_msg_id])
            return

        compiled_traps = compile_patterns(mapping.get('trap_phrases', []))
        if compiled_traps and compiled_traps.search(text_lower):
            reason = "Block phrase in edited text"
            await notify_trap(event, mapping, pair_name, reason)
            await client.delete_messages(int(mapping['destination']), [forwarded_msg_id])
            return

        processed_text = message_text
        processed_entities = original_entities
        if message_text:
            processed_text = normalized_text
            processed_text = remove_patterns(processed_text, mapping.get('header_patterns', []))
            processed_text = remove_patterns(processed_text, mapping.get('footer_patterns', []))
            processed_text, phrases_removed = remove_phrases(processed_text, mapping.get('remove_phrases', []))
            if phrases_removed:
                processed_text = paraphrase_phrases(processed_text, mapping.get('remove_phrases', []))
            if mapping.get('remove_mentions', False):
                processed_text = re.sub(r'@[a-zA-Z0-9_]+|t\.me/[^\s]+', '', processed_text)
                processed_text = re.sub(r'\s+', ' ', processed_text).strip()
            processed_text = apply_custom_header_footer(
                processed_text, mapping.get('custom_header', ''), mapping.get('custom_footer', '')
            )
            if normalize_text(processed_text).strip().lower() != normalize_text(message_text).strip().lower():
                processed_entities = adjust_entities(message_text, processed_text, original_entities)
                logger.info("Text changed, adjusted entities.")

        processed_media = None
        if isinstance(media, MessageMediaPhoto):
            processed_media, is_trapped = await process_image(event, mapping)
            if is_trapped:
                await client.delete_messages(int(mapping['destination']), [forwarded_msg_id])
                return
            if not processed_media:
                logger.warning(f"Failed to process edited image for pair '{pair_name}'. Using original media.")
                processed_media = media
        elif isinstance(media, MessageMediaDocument):
            processed_media = media

        if not processed_text.strip() and not processed_media:
            await client.delete_messages(int(mapping['destination']), [forwarded_msg_id])
            return

        await client.edit_message(
            entity=int(mapping['destination']),
            message=forwarded_msg_id,
            text=processed_text,
            file=processed_media,
            parse_mode='html',
            formatting_entities=processed_entities
        )
        pair_stats[user_id][pair_name]['edited'] += 1
        pair_stats[user_id][pair_name]['last_activity'] = datetime.now().isoformat()
        logger.info(f"Edited copied message {forwarded_msg_id} in {mapping['destination']}")
    except errors.MessageIdInvalidError:
        logger.warning(f"Invalid message ID {forwarded_msg_id} for pair '{pair_name}'. Skipping edit.")
    except Exception as e:
        logger.error(f"Error editing message for pair '{pair_name}': {e}")

async def delete_copied_message(
    event: events.MessageDeleted.Event,
    mapping: Dict[str, Any],
    user_id: str,
    pair_name: str
):
    """Delete copied message in destination channel."""
    try:
        for msg_id in event.deleted_ids:
            mapping_key = f"{mapping['source']}:{msg_id}"
            if mapping_key in client.forwarded_messages:
                forwarded_msg_id = client.forwarded_messages[mapping_key]
                await client.delete_messages(int(mapping['destination']), [forwarded_msg_id])
                pair_stats[user_id][pair_name]['deleted'] += 1
                pair_stats[user_id][pair_name]['last_activity'] = datetime.now().isoformat()
                del client.forwarded_messages[mapping_key]
                logger.info(f"Deleted copied message {forwarded_msg_id} in {mapping['destination']}")
    except errors.MessageIdInvalidError:
        logger.warning(f"Invalid message ID {forwarded_msg_id} for pair '{pair_name}'. Skipping deletion.")
    except Exception as e:
        logger.error(f"Error deleting copied message for pair '{pair_name}': {e}")

async def handle_reply_mapping(event: events.NewMessage.Event, mapping: Dict[str, Any]) -> Optional[int]:
    """Handle reply message mapping."""
    if not hasattr(event.message, 'reply_to') or not event.message.reply_to:
        return None
    try:
        source_reply_id = event.message.reply_to.reply_to_msg_id
        mapping_key = f"{mapping['source']}:{source_reply_id}"
        if mapping_key in client.forwarded_messages:
            return client.forwarded_messages[mapping_key]
        logger.info(f"No forwarded message found for reply ID {source_reply_id} in pair '{mapping.get('pair_name', 'unknown')}'")
        return None
    except Exception as e:
        logger.error(f"Error handling reply mapping for pair '{mapping.get('pair_name', 'unknown')}': {e}")
        return None

async def store_message_mapping(event: events.NewMessage.Event, mapping: Dict[str, Any], sent_message: Any):
    """Store message mapping for tracking."""
    try:
        if not hasattr(event.message, 'id'):
            return
        mapping_key = f"{mapping['source']}:{event.message.id}"
        if len(client.forwarded_messages) >= MAX_MAPPING_HISTORY:
            oldest_key = next(iter(client.forwarded_messages))
            client.forwarded_messages.pop(oldest_key)
        client.forwarded_messages[mapping_key] = sent_message.id
        logger.info(f"Stored message mapping for pair '{mapping.get('pair_name', 'unknown')}': {mapping_key} -> {sent_message.id}")
    except Exception as e:
        logger.error(f"Error storing message mapping for pair '{mapping.get('pair_name', 'unknown')}': {e}")

def apply_custom_header_footer(text: str, header: str, footer: str) -> str:
    """Apply custom header and footer to text."""
    if not text:
        return text
    result = text
    if header:
        result = header + '\n' + result
    if footer:
        result = result + '\n' + footer
    return result.strip()

async def process_queue():
    """Process messages from the queue concurrently."""
    while not shutdown_event.is_set():
        try:
            if message_queue:
                event, mapping, user_id, pair_name, queued_time = message_queue.popleft()
                async with queue_semaphore:
                    await copy_message_with_retry(event, mapping, user_id, pair_name)
            else:
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Queue processing error: {e}")
            await asyncio.sleep(1)

@client.on(events.NewMessage)
async def copy_messages(event: events.NewMessage.Event):
    """Handle new messages for copying."""
    try:
        if not is_processing_enabled or not is_connected:
            return
        destinations = []
        for user_id, pairs in channel_mappings.items():
            for pair_name, mapping in pairs.items():
                if mapping['status'] != 'active':
                    continue
                source_id = str(event.chat_id)
                if source_id == mapping['source']:
                    mapping['pair_name'] = pair_name
                    destinations.append((event, mapping, user_id, pair_name, datetime.now()))
        random.shuffle(destinations)  # Randomize post order for stealth
        for dest in destinations:
            if len(message_queue) >= MAX_QUEUE_SIZE:
                logger.warning(f"Queue full, dropping message for pair '{dest[3]}'")
                continue
            message_queue.append(dest)
            pair_stats[dest[2]][dest[3]]['queued'] += 1
            logger.debug(f"Queued message for pair '{dest[3]}'")
    except Exception as e:
        logger.error(f"Error in copy_messages handler: {e}")

@client.on(events.MessageEdited)
async def handle_message_edit(event: events.MessageEdited.Event):
    """Handle edited messages."""
    try:
        if not is_processing_enabled or not is_connected:
            return
        for user_id, pairs in channel_mappings.items():
            for pair_name, mapping in pairs.items():
                if mapping['status'] != 'active':
                    continue
                source_id = str(event.chat_id)
                if source_id == mapping['source']:
                    mapping['pair_name'] = pair_name
                    await edit_copied_message(event, mapping, user_id, pair_name)
    except Exception as e:
        logger.error(f"Error in handle_message_edit: {e}")

@client.on(events.MessageDeleted)
async def handle_message_delete(event: events.MessageDeleted.Event):
    """Handle deleted messages."""
    try:
        if not is_processing_enabled or not is_connected:
            return
        for user_id, pairs in channel_mappings.items():
            for pair_name, mapping in pairs.items():
                if mapping['status'] != 'active':
                    continue
                source_id = str(event.chat_id)
                if source_id == mapping['source']:
                    mapping['pair_name'] = pair_name
                    await delete_copied_message(event, mapping, user_id, pair_name)
    except Exception as e:
        logger.error(f"Error in handle_message_delete: {e}")

@client.on(events.NewMessage(pattern='(?i)^/commands$'))
async def list_commands(event: events.NewMessage.Event):
    """List available bot commands."""
    global OWNER_ID
    if OWNER_ID is None:
        OWNER_ID = event.sender_id
        logger.info(f"Set OWNER_ID to {OWNER_ID} via /commands")
    try:
        commands = """
🚀 GhostCopyBotPro Commands

**Setup & Management**
- `/setowner <user_id>` - Set owner ID for notifications
- `/setpair <name> <source> <dest> [yes|no]` - Add pair (yes/no for mentions)
- `/listpairs` - Show all pairs
- `/pausepair <name>` - Pause a pair
- `/resumepair <name>` - Resume a pair
- `/pauseall` - Pause all pairs
- `/resumeall` - Resume all pairs
- `/clearpairs` - Remove all pairs
- `/setdelay <name> <min> <max> [offset]` - Set delay range and offset
- `/status <name>` - Check pair status
- `/report` - View pair stats
- `/monitor` - Detailed pair monitor
- `/healthcheck` - Check session load
- `/flushcache` - Clear image and reply mappings
- `/stealthcheck` - Simulate stealth repost

**🧽 Text Cleaning**
- `/addheader <pair> <pattern>` - Add header to remove
- `/removeheader <pair> <pattern>` - Remove header
- `/addfooter <pair> <pattern>` - Add footer to remove
- `/removefooter <pair> <pattern>` - Remove footer
- `/addremoveword <pair> <phrase>` - Add phrase to remove
- `/removeword <pair> <phrase>` - Remove phrase
- `/enablementionremoval <pair>` - Enable mention removal
- `/disablementionremoval <pair>` - Disable mention removal
- `/showfilters <pair>` - Show text filters
- `/setcustomheader <pair> <text>` - Set custom header
- `/setcustomfooter <pair> <text>` - Set custom footer
- `/clearcustomheaderfooter <pair>` - Clear custom text
- `/addhoneypot <pair> <phrase>` - Add honeypot phrase
- `/removehoneypot <pair> <phrase>` - Remove honeypot phrase
- `/showhoneypots <pair>` - Show honeypot phrases

**🚫 Block Filters**
- `/addblockword <pair> <word>` - Add block phrase
- `/removeblockword <pair> <word>` - Remove block phrase
- `/addblockimage <pair>` - Add block image (reply to image)
- `/removeblockimage <pair> <hash>` - Remove block image hash
- `/showblocks <pair>` - Show block filters
"""
        await event.reply(commands)
        logger.info(f"Bot commands listed by user {event.sender_id}")
    except Exception as e:
        logger.error(f"Error listing commands: {e}")
        await event.reply("❌ Failed to list commands.")

@client.on(events.NewMessage(pattern=r'/setowner (\d+)'))
async def set_owner(event: events.NewMessage.Event):
    """Set OWNER_ID for notifications."""
    global OWNER_ID
    try:
        new_owner_id = int(event.pattern_match.group(1))
        OWNER_ID = new_owner_id
        save_mappings()
        await event.reply(f"✅ Set OWNER_ID to {new_owner_id}.")
        logger.info(f"Set OWNER_ID to {new_owner_id} by user {event.sender_id}")
    except ValueError:
        await event.reply("❌ Invalid user ID format.")
    except Exception as e:
        logger.error(f"Error setting OWNER_ID: {e}")
        await event.reply("❌ Failed to set OWNER_ID.")

@client.on(events.NewMessage(pattern=r'/stealthcheck'))
async def stealth_check(event: events.NewMessage.Event):
    """Simulate a stealth repost to test filters and processing."""
    try:
        if not event.message.reply_to:
            await event.reply("🚀 Please reply to a message to test stealth repost.")
            return
        replied_msg = await event.get_reply_message()
        user_id = str(event.sender_id)
        pair_name = "stealth_test"
        mapping = {
            'source': str(event.chat_id),
            'destination': str(event.chat_id),
            'pair_name': pair_name,
            'status': 'active',
            'remove_mentions': True,
            'header_patterns': [],
            'footer_patterns': [],
            'remove_phrases': [],
            'trap_phrases': [],
            'honeypot_phrases': ['do_not_copy'],
            'trap_image_hashes': [],
            'delay_range': [0, 0],
            'delay_offset': 0,
            'custom_header': '',
            'custom_footer': ''
        }

        message_text = replied_msg.raw_text or ""
        normalized_text = normalize_text(message_text)
        text_lower = normalized_text.lower()
        original_entities = replied_msg.entities or []
        media = replied_msg.media

        compiled_honeypots = compile_patterns(mapping.get('honeypot_phrases', []))
        if compiled_honeypots and compiled_honeypots.search(text_lower):
            await event.reply("🚫 Message contains honeypot phrase, would be skipped.")
            return

        compiled_traps = compile_patterns(mapping.get('trap_phrases', []))
        if compiled_traps and compiled_traps.search(text_lower):
            await event.reply("🚫 Message contains trap phrase, would be blocked.")
            return

        processed_text = message_text
        processed_entities = original_entities
        if message_text:
            processed_text = normalized_text
            processed_text = remove_patterns(processed_text, mapping.get('header_patterns', []))
            processed_text = remove_patterns(processed_text, mapping.get('footer_patterns', []))
            processed_text, phrases_removed = remove_phrases(processed_text, mapping.get('remove_phrases', []))
            if phrases_removed:
                processed_text = paraphrase_phrases(processed_text, mapping.get('remove_phrases', []))
            if mapping.get('remove_mentions', False):
                processed_text = re.sub(r'@[a-zA-Z0-9_]+|t\.me/[^\s]+', '', processed_text)
                processed_text = re.sub(r'\s+', ' ', processed_text).strip()
            processed_text = apply_custom_header_footer(
                processed_text, mapping.get('custom_header', ''), mapping.get('custom_footer', '')
            )
            if normalize_text(processed_text).strip().lower() != normalize_text(message_text).strip().lower():
                processed_entities = adjust_entities(message_text, processed_text, original_entities)

        processed_media = None
        if isinstance(media, MessageMediaPhoto):
            processed_media, is_trapped = await process_image(replied_msg, mapping)
            if is_trapped:
                await event.reply("🚫 Image contains trap content, would be blocked.")
                return
            if not processed_media:
                logger.warning(f"Failed to process image for stealth check.")
                processed_media = media

        if not processed_text.strip() and not processed_media:
            await event.reply("🚫 Message empty after filtering, would be blocked.")
            return

        await client.send_message(
            event.chat_id,
            file=processed_media,
            message=processed_text,
            reply_to=replied_msg.id,
            parse_mode='html',
            formatting_entities=processed_entities
        )
        await event.reply("🚀 Stealth repost simulation successful. Check the processed message above.")
        logger.info(f"Stealth check performed by user {event.sender_id}")
    except Exception as e:
        logger.error(f"Error in stealth check: {e}")
        await event.reply("❌ Failed to perform stealth check.")

@client.on(events.NewMessage(pattern=r'/healthcheck'))
async def health_check(event: events.NewMessage.Event):
    """Check session load."""
    try:
        pair_count = sum(len(pairs) for pairs in channel_mappings.values())
        health_report = [
            f"🚀 Health Check:",
            f"Active Pairs: {pair_count}/{MAX_CHATS_PER_SESSION}",
            f"Queue Size: {len(message_queue)}/{MAX_QUEUE_SIZE}",
            f"Connected: {'Yes' if is_connected else 'No'}",
            f"Processing Enabled: {'Yes' if is_processing_enabled else 'No'}",
            f"Owner ID Set: {'Yes' if OWNER_ID else 'No'}"
        ]
        await event.reply("\n".join(health_report))
        logger.info(f"Health check performed by user {event.sender_id}")
    except Exception as e:
        logger.error(f"Error performing health check: {e}")
        await event.reply("❌ Failed to perform health check.")

@client.on(events.NewMessage(pattern=r'/flushcache'))
async def flush_cache(event: events.NewMessage.Event):
    """Clear image and reply mappings cache."""
    try:
        async with aiosqlite.connect(CACHE_DB_PATH) as db:
            await db.execute("DELETE FROM image_cache")
            await db.commit()
        client.forwarded_messages.clear()
        await event.reply("🧹 Image and reply mappings cache cleared.")
        logger.info(f"Cache flushed by user {event.sender_id}")
    except Exception as e:
        logger.error(f"Error flushing cache: {e}")
        await event.reply("❌ Failed to flush cache.")

@client.on(events.NewMessage(pattern=r'/setpair (\S+) (\S+) (\S+)(?: (yes|no))?'))
async def set_pair(event: events.NewMessage.Event):
    """Set up a new source-destination pair."""
    try:
        pair_name, source, destination, remove_mentions = event.pattern_match.groups()
        user_id = str(event.sender_id)
        pair_name = pair_name.strip()
        remove_mentions = remove_mentions == "yes"
        total_pairs = sum(len(pairs) for pairs in channel_mappings.values())
        if total_pairs >= MAX_CHATS_PER_SESSION:
            await event.reply(f"❌ Maximum number of pairs ({MAX_CHATS_PER_SESSION}) reached.")
            return
        if user_id not in channel_mappings:
            channel_mappings[user_id] = {}
        if user_id not in pair_stats:
            pair_stats[user_id] = {}
        channel_mappings[user_id][pair_name] = {
            'source': source.strip(),
            'destination': destination.strip(),
            'status': 'active',
            'remove_mentions': remove_mentions,
            'header_patterns': [],
            'footer_patterns': [],
            'remove_phrases': [],
            'trap_phrases': [],
            'honeypot_phrases': ['do_not_copy', 'trapword'],
            'trap_image_hashes': [],
            'delay_range': DEFAULT_DELAY_RANGE,
            'delay_offset': DEFAULT_DELAY_OFFSET,
            'custom_header': '',
            'custom_footer': '',
            'last_activity': None
        }
        pair_stats[user_id][pair_name] = {'forwarded': 0, 'edited': 0, 'deleted': 0, 'blocked': 0, 'queued': 0, 'last_activity': None}
        save_mappings()
        await event.reply(f"✅ Pair '{pair_name}' added: {source} ➡️ {destination}\nMentions removal: {'✅' if remove_mentions else '❌'}")
        logger.info(f"Set pair '{pair_name}' for user {user_id}: {source} -> {destination}")
    except Exception as e:
        logger.error(f"Error setting pair: {e}")
        await event.reply("❌ Failed to set pair.")

@client.on(events.NewMessage(pattern=r'/listpairs'))
async def list_pairs(event: events.NewMessage.Event):
    """List all channel pairs."""
    try:
        user_id = str(event.sender_id)
        if user_id not in channel_mappings or not channel_mappings[user_id]:
            await event.reply("❌ No pairs configured.")
            return
        pairs_info = [f"📜 Pairs for user {user_id}:"]
        for pair_name, mapping in channel_mappings[user_id].items():
            status = mapping['status']
            pairs_info.append(
                f"Pair: {pair_name} | Source: {mapping['source']} ➡️ Dest: {mapping['destination']} | "
                f"Status: {status} | Mentions: {'✅' if mapping['remove_mentions'] else '❌'}"
            )
        await event.reply("\n".join(pairs_info))
        logger.info(f"Listed pairs for user {user_id}")
    except Exception as e:
        logger.error(f"Error listing pairs: {e}")
        await event.reply("❌ Failed to list pairs.")

@client.on(events.NewMessage(pattern=r'/pausepair (\S+)'))
async def pause_pair(event: events.NewMessage.Event):
    """Pause a specific pair."""
    try:
        pair_name = event.pattern_match.group(1).strip()
        user_id = str(event.sender_id)
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        channel_mappings[user_id][pair_name]['status'] = 'paused'
        save_mappings()
        await event.reply(f"⏸️ Pair '{pair_name}' paused.")
        logger.info(f"Paused pair '{pair_name}' for user {user_id}")
    except Exception as e:
        logger.error(f"Error pausing pair: {e}")
        await event.reply("❌ Failed to pause pair.")

@client.on(events.NewMessage(pattern=r'/resumepair (\S+)'))
async def resume_pair(event: events.NewMessage.Event):
    """Resume a specific pair."""
    try:
        pair_name = event.pattern_match.group(1).strip()
        user_id = str(event.sender_id)
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        channel_mappings[user_id][pair_name]['status'] = 'active'
        save_mappings()
        await event.reply(f"▶️ Pair '{pair_name}' resumed.")
        logger.info(f"Resumed pair '{pair_name}' for user {user_id}")
    except Exception as e:
        logger.error(f"Error resuming pair: {e}")
        await event.reply("❌ Failed to resume pair.")

@client.on(events.NewMessage(pattern=r'/pauseall'))
async def pause_all(event: events.NewMessage.Event):
    """Pause all pairs."""
    try:
        user_id = str(event.sender_id)
        if user_id not in channel_mappings:
            await event.reply("❌ No pairs configured.")
            return
        for pair_name in channel_mappings[user_id]:
            channel_mappings[user_id][pair_name]['status'] = 'paused'
        save_mappings()
        await event.reply("⏸️ All pairs paused.")
        logger.info(f"Paused all pairs for user {user_id}")
    except Exception as e:
        logger.error(f"Error pausing all pairs: {e}")
        await event.reply("❌ Failed to pause all pairs.")

@client.on(events.NewMessage(pattern=r'/resumeall'))
async def resume_all(event: events.NewMessage.Event):
    """Resume all pairs."""
    try:
        user_id = str(event.sender_id)
        if user_id not in channel_mappings:
            await event.reply("❌ No pairs configured.")
            return
        for pair_name in channel_mappings[user_id]:
            channel_mappings[user_id][pair_name]['status'] = 'active'
        save_mappings()
        await event.reply("▶️ All pairs resumed.")
        logger.info(f"Resumed all pairs for user {user_id}")
    except Exception as e:
        logger.error(f"Error resuming all pairs: {e}")
        await event.reply("❌ Failed to resume all pairs.")

@client.on(events.NewMessage(pattern=r'/clearpairs'))
async def clear_pairs(event: events.NewMessage.Event):
    """Clear all pairs."""
    try:
        user_id = str(event.sender_id)
        if user_id in channel_mappings:
            del channel_mappings[user_id]
            del pair_stats[user_id]
            save_mappings()
            await event.reply("🧹 All pairs cleared.")
            logger.info(f"Cleared all pairs for user {user_id}")
        else:
            await event.reply("❌ No pairs to clear.")
    except Exception as e:
        logger.error(f"Error clearing pairs: {e}")
        await event.reply("❌ Failed to clear pairs.")

@client.on(events.NewMessage(pattern=r'/setdelay (\S+) (\d+\.?\d*) (\d+\.?\d*) ?(\d+\.?\d*)?'))
async def set_delay(event: events.NewMessage.Event):
    """Set delay range and offset for a pair."""
    try:
        pair_name, min_delay, max_delay, offset = event.pattern_match.groups()
        user_id = str(event.sender_id)
        pair_name = pair_name.strip()
        min_delay = float(min_delay)
        max_delay = float(max_delay)
        offset = float(offset) if offset else DEFAULT_DELAY_OFFSET
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        if min_delay > max_delay:
            await event.reply("❌ Min delay must be less than or equal to max delay.")
            return
        channel_mappings[user_id][pair_name]['delay_range'] = [min_delay, max_delay]
        channel_mappings[user_id][pair_name]['delay_offset'] = offset
        save_mappings()
        await event.reply(f"⏱️ Set delay for '{pair_name}': {min_delay}s–{max_delay}s, offset {offset}s")
        logger.info(f"Set delay for pair '{pair_name}' for user {user_id}: {min_delay}–{max_delay}s, offset {offset}s")
    except Exception as e:
        logger.error(f"Error setting delay: {e}")
        await event.reply("❌ Failed to set delay.")

@client.on(events.NewMessage(pattern=r'/status (\S+)'))
async def check_status(event: events.NewMessage.Event):
    """Check status of a pair."""
    try:
        pair_name = event.pattern_match.group(1).strip()
        user_id = str(event.sender_id)
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        mapping = channel_mappings[user_id][pair_name]
        stats = pair_stats[user_id][pair_name]
        status_info = [
            f"📊 Status for pair '{pair_name}':",
            f"Source: {mapping['source']}",
            f"Destination: {mapping['destination']}",
            f"Status: {mapping['status']}",
            f"Last Activity: {stats['last_activity'] or 'None'}",
            f"Forwarded: {stats['forwarded']}",
            f"Edited: {stats['edited']}",
            f"Deleted: {stats['deleted']}",
            f"Blocked: {stats['blocked']}",
            f"Queued: {stats['queued']}"
        ]
        await event.reply("\n".join(status_info))
        logger.info(f"Checked status for pair '{pair_name}' for user {user_id}")
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        await event.reply("❌ Failed to check status.")

@client.on(events.NewMessage(pattern=r'/report'))
async def report(event: events.NewMessage.Event):
    """View stats for all pairs."""
    try:
        user_id = str(event.sender_id)
        if user_id not in channel_mappings:
            await event.reply("❌ No pairs configured.")
            return
        report_info = [f"📈 Report for user {user_id}:"]
        for pair_name, stats in pair_stats[user_id].items():
            mapping = channel_mappings[user_id][pair_name]
            report_info.append(
                f"Pair: {pair_name} | Status: {mapping['status']} | "
                f"Forwarded: {stats['forwarded']} | Blocked: {stats['blocked']} | "
                f"Last: {stats['last_activity'] or 'None'}"
            )
        await event.reply("\n".join(report_info))
        logger.info(f"Generated report for user {user_id}")
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        await event.reply("❌ Failed to generate report.")

@client.on(events.NewMessage(pattern=r'/monitor'))
async def monitor(event: events.NewMessage.Event):
    """Detailed monitor of all pairs."""
    try:
        user_id = str(event.sender_id)
        if user_id not in channel_mappings:
            await event.reply("❌ No pairs configured.")
            return
        monitor_info = [f"🔍 Monitor for user {user_id}:"]
        for pair_name, mapping in channel_mappings[user_id].items():
            stats = pair_stats[user_id][pair_name]
            monitor_info.append(
                f"Pair: {pair_name}\n"
                f"Source: {mapping['source']} ➡️ Dest: {mapping['destination']}\n"
                f"Status: {mapping['status']} | Mentions: {'✅' if mapping['remove_mentions'] else '❌'}\n"
                f"Delay: {mapping['delay_range'][0]}–{mapping['delay_range'][1]}s, offset {mapping['delay_offset']}s\n"
                f"Forwarded: {stats['forwarded']} | Edited: {stats['edited']} | Deleted: {stats['deleted']} | "
                f"Blocked: {stats['blocked']} | Queued: {stats['queued']}\n"
                f"Last Activity: {stats['last_activity'] or 'None'}\n"
            )
        await send_split_message(client, event.chat_id, "\n".join(monitor_info))
        logger.info(f"Generated monitor for user {user_id}")
    except Exception as e:
        logger.error(f"Error generating monitor: {e}")
        await event.reply("❌ Failed to generate monitor.")

@client.on(events.NewMessage(pattern=r'/addheader (\S+) (.+)'))
async def add_header(event: events.NewMessage.Event):
    """Add header pattern to remove."""
    try:
        pair_name, pattern = event.pattern_match.groups()
        user_id = str(event.sender_id)
        pair_name = pair_name.strip()
        pattern = pattern.strip()
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        if pattern not in channel_mappings[user_id][pair_name]['header_patterns']:
            channel_mappings[user_id][pair_name]['header_patterns'].append(pattern)
            save_mappings()
            await event.reply(f"🧽 Added header pattern for '{pair_name}': {pattern}")
            logger.info(f"Added header pattern '{pattern}' to pair '{pair_name}' for user {user_id}")
        else:
            await event.reply(f"⏸️ Header pattern already exists in '{pair_name}'.")
    except Exception as e:
        logger.error(f"Error adding header: {e}")
        await event.reply("❌ Failed to add header.")

@client.on(events.NewMessage(pattern=r'/removeheader (\S+) (.+)'))
async def remove_header(event: events.NewMessage.Event):
    """Remove header pattern."""
    try:
        pair_name, pattern = event.pattern_match.groups()
        user_id = str(event.sender_id)
        pair_name = pair_name.strip()
        pattern = pattern.strip()
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        if pattern in channel_mappings[user_id][pair_name]['header_patterns']:
            channel_mappings[user_id][pair_name]['header_patterns'].remove(pattern)
            save_mappings()
            await event.reply(f"🧽 Removed header pattern from '{pair_name}': {pattern}")
            logger.info(f"Removed header pattern '{pattern}' from pair '{pair_name}' for user {user_id}")
        else:
            await event.reply(f"❌ Header pattern not found in '{pair_name}'.")
    except Exception as e:
        logger.error(f"Error removing header: {e}")
        await event.reply("❌ Failed to remove header.")

@client.on(events.NewMessage(pattern=r'/addfooter (\S+) (.+)'))
async def add_footer(event: events.NewMessage.Event):
    """Add footer pattern to remove."""
    try:
        pair_name, pattern = event.pattern_match.groups()
        user_id = str(event.sender_id)
        pair_name = pair_name.strip()
        pattern = pattern.strip()
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        if pattern not in channel_mappings[user_id][pair_name]['footer_patterns']:
            channel_mappings[user_id][pair_name]['footer_patterns'].append(pattern)
            save_mappings()
            await event.reply(f"🧽 Added footer pattern for '{pair_name}': {pattern}")
            logger.info(f"Added footer pattern '{pattern}' to pair '{pair_name}' for user {user_id}")
        else:
            await event.reply(f"⏸️ Footer pattern already exists in '{pair_name}'.")
    except Exception as e:
        logger.error(f"Error adding footer: {e}")
        await event.reply("❌ Failed to add footer.")

@client.on(events.NewMessage(pattern=r'/removefooter (\S+) (.+)'))
async def remove_footer(event: events.NewMessage.Event):
    """Remove footer pattern."""
    try:
        pair_name, pattern = event.pattern_match.groups()
        user_id = str(event.sender_id)
        pair_name = pair_name.strip()
        pattern = pattern.strip()
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        if pattern in channel_mappings[user_id][pair_name]['footer_patterns']:
            channel_mappings[user_id][pair_name]['footer_patterns'].remove(pattern)
            save_mappings()
            await event.reply(f"🧽 Removed footer pattern from '{pair_name}': {pattern}")
            logger.info(f"Removed footer pattern '{pattern}' from pair '{pair_name}' for user {user_id}")
        else:
            await event.reply(f"❌ Footer pattern not found in '{pair_name}'.")
    except Exception as e:
        logger.error(f"Error removing footer: {e}")
        await event.reply("❌ Failed to remove footer.")

@client.on(events.NewMessage(pattern=r'/addremoveword (\S+) (.+)'))
async def add_remove_word(event: events.NewMessage.Event):
    """Add phrase to remove."""
    try:
        pair_name, phrase = event.pattern_match.groups()
        user_id = str(event.sender_id)
        pair_name = pair_name.strip()
        phrase = phrase.strip()
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        if phrase not in channel_mappings[user_id][pair_name]['remove_phrases']:
            channel_mappings[user_id][pair_name]['remove_phrases'].append(phrase)
            save_mappings()
            await event.reply(f"🧽 Added phrase to remove for '{pair_name}': {phrase}")
            logger.info(f"Added remove phrase '{phrase}' to pair '{pair_name}' for user {user_id}")
        else:
            await event.reply(f"⏸️ Phrase already exists in '{pair_name}'.")
    except Exception as e:
        logger.error(f"Error adding remove word: {e}")
        await event.reply("❌ Failed to add remove word.")

@client.on(events.NewMessage(pattern=r'/removeword (\S+) (.+)'))
async def remove_word(event: events.NewMessage.Event):
    """Remove phrase from removal list."""
    try:
        pair_name, phrase = event.pattern_match.groups()
        user_id = str(event.sender_id)
        pair_name = pair_name.strip()
        phrase = phrase.strip()
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        if phrase in channel_mappings[user_id][pair_name]['remove_phrases']:
            channel_mappings[user_id][pair_name]['remove_phrases'].remove(phrase)
            save_mappings()
            await event.reply(f"🧽 Removed phrase from '{pair_name}': {phrase}")
            logger.info(f"Removed phrase '{phrase}' from pair '{pair_name}' for user {user_id}")
        else:
            await event.reply(f"❌ Phrase not found in '{pair_name}'.")
    except Exception as e:
        logger.error(f"Error removing phrase: {e}")
        await event.reply("❌ Failed to remove phrase.")

@client.on(events.NewMessage(pattern=r'/enablementionremoval (\S+)'))
async def enable_mention_removal(event: events.NewMessage.Event):
    """Enable mention removal for a pair."""
    try:
        pair_name = event.pattern_match.group(1).strip()
        user_id = str(event.sender_id)
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        channel_mappings[user_id][pair_name]['remove_mentions'] = True
        save_mappings()
        await event.reply(f"✅ Mention removal enabled for '{pair_name}'.")
        logger.info(f"Enabled mention removal for pair '{pair_name}' for user {user_id}")
    except Exception as e:
        logger.error(f"Error enabling mention removal: {e}")
        await event.reply("❌ Failed to enable mention removal.")

@client.on(events.NewMessage(pattern=r'/disablementionremoval (\S+)'))
async def disable_mention_removal(event: events.NewMessage.Event):
    """Disable mention removal for a pair."""
    try:
        pair_name = event.pattern_match.group(1).strip()
        user_id = str(event.sender_id)
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        channel_mappings[user_id][pair_name]['remove_mentions'] = False
        save_mappings()
        await event.reply(f"❌ Mention removal disabled for '{pair_name}'.")
        logger.info(f"Disabled mention removal for pair '{pair_name}' for user {user_id}")
    except Exception as e:
        logger.error(f"Error disabling mention removal: {e}")
        await event.reply("❌ Failed to disable mention removal.")

@client.on(events.NewMessage(pattern=r'/showfilters (\S+)'))
async def show_filters(event: events.NewMessage.Event):
    """Show text filters for a pair."""
    try:
        pair_name = event.pattern_match.group(1).strip()
        user_id = str(event.sender_id)
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        mapping = channel_mappings[user_id][pair_name]
        filters_info = [
            f"🧽 Filters for pair '{pair_name}':",
            f"Headers: {', '.join(mapping['header_patterns']) or 'None'}",
            f"Footers: {', '.join(mapping['footer_patterns']) or 'None'}",
            f"Remove Phrases: {', '.join(mapping['remove_phrases']) or 'None'}",
            f"Mention Removal: {'✅' if mapping['remove_mentions'] else '❌'}",
            f"Custom Header: {mapping['custom_header'] or 'None'}",
            f"Custom Footer: {mapping['custom_footer'] or 'None'}"
        ]
        await event.reply("\n".join(filters_info))
        logger.info(f"Showed filters for pair '{pair_name}' for user {user_id}")
    except Exception as e:
        logger.error(f"Error showing filters: {e}")
        await event.reply("❌ Failed to show filters.")

@client.on(events.NewMessage(pattern=r'/setcustomheader (\S+) (.+)'))
async def set_custom_header(event: events.NewMessage.Event):
    """Set custom header for a pair."""
    try:
        pair_name, header = event.pattern_match.groups()
        user_id = str(event.sender_id)
        pair_name = pair_name.strip()
        header = header.strip()
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        channel_mappings[user_id][pair_name]['custom_header'] = header
        save_mappings()
        await event.reply(f"🧽 Set custom header for '{pair_name}': {header}")
        logger.info(f"Set custom header '{header}' for pair '{pair_name}' for user {user_id}")
    except Exception as e:
        logger.error(f"Error setting custom header: {e}")
        await event.reply("❌ Failed to set custom header.")

@client.on(events.NewMessage(pattern=r'/setcustomfooter (\S+) (.+)'))
async def set_custom_footer(event: events.NewMessage.Event):
    """Set custom footer for a pair."""
    try:
        pair_name, footer = event.pattern_match.groups()
        user_id = str(event.sender_id)
        pair_name = pair_name.strip()
        footer = footer.strip()
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        channel_mappings[user_id][pair_name]['custom_footer'] = footer
        save_mappings()
        await event.reply(f"🧽 Set custom footer for '{pair_name}': {footer}")
        logger.info(f"Set custom footer '{footer}' for pair '{pair_name}' for user {user_id}")
    except Exception as e:
        logger.error(f"Error setting custom footer: {e}")
        await event.reply("❌ Failed to set custom footer.")

@client.on(events.NewMessage(pattern=r'/clearcustomheaderfooter (\S+)'))
async def clear_custom_header_footer(event: events.NewMessage.Event):
    """Clear custom header and footer for a pair."""
    try:
        pair_name = event.pattern_match.group(1).strip()
        user_id = str(event.sender_id)
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        channel_mappings[user_id][pair_name]['custom_header'] = ''
        channel_mappings[user_id][pair_name]['custom_footer'] = ''
        save_mappings()
        await event.reply(f"🧹 Cleared custom header and footer for '{pair_name}'.")
        logger.info(f"Cleared custom header/footer for pair '{pair_name}' for user {user_id}")
    except Exception as e:
        logger.error(f"Error clearing custom header/footer: {e}")
        await event.reply("❌ Failed to clear custom header/footer.")

@client.on(events.NewMessage(pattern=r'/addblockword (\S+) (.+)'))
async def add_block_word(event: events.NewMessage.Event):
    """Add a trap phrase to block."""
    try:
        pair_name, phrase = event.pattern_match.groups()
        user_id = str(event.sender_id)
        pair_name = pair_name.strip()
        phrase = phrase.strip()
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        if phrase not in channel_mappings[user_id][pair_name]['trap_phrases']:
            channel_mappings[user_id][pair_name]['trap_phrases'].append(phrase)
            save_mappings()
            await event.reply(f"🚫 Added block phrase for '{pair_name}': {phrase}")
            logger.info(f"Added block phrase '{phrase}' to pair '{pair_name}' for user {user_id}")
        else:
            await event.reply(f"⏸️ Block phrase already exists in '{pair_name}'.")
    except Exception as e:
        logger.error(f"Error adding block word: {e}")
        await event.reply("❌ Failed to add block word.")

@client.on(events.NewMessage(pattern=r'/removeblockword (\S+) (.+)'))
async def remove_block_word(event: events.NewMessage.Event):
    """Remove a trap phrase from block list."""
    try:
        pair_name, phrase = event.pattern_match.groups()
        user_id = str(event.sender_id)
        pair_name = pair_name.strip()
        phrase = phrase.strip()
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        if phrase in channel_mappings[user_id][pair_name]['trap_phrases']:
            channel_mappings[user_id][pair_name]['trap_phrases'].remove(phrase)
            save_mappings()
            await event.reply(f"🚫 Removed block phrase from '{pair_name}': {phrase}")
            logger.info(f"Removed block phrase '{phrase}' from pair '{pair_name}' for user {user_id}")
        else:
            await event.reply(f"❌ Block phrase not found in '{pair_name}'.")
    except Exception as e:
        logger.error(f"Error removing block word: {e}")
        await event.reply("❌ Failed to remove block word.")

@client.on(events.NewMessage(pattern=r'/addblockimage (\S+)'))
async def add_block_image(event: events.NewMessage.Event):
    """Add an image’s perceptual hash to block list."""
    try:
        pair_name = event.pattern_match.group(1).strip()
        user_id = str(event.sender_id)
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        if not event.message.reply_to or not isinstance(event.message.reply_to.media, MessageMediaPhoto):
            await event.reply("🖼️ Please reply to a photo to block.")
            return
        replied_msg = await event.get_reply_message()
        photo = await client.download_media(replied_msg, bytes)
        if not photo:
            await event.reply("❌ Failed to download photo.")
            return
        image = Image.open(io.BytesIO(photo))
        phash = str(imagehash.phash(image))
        if phash not in channel_mappings[user_id][pair_name]['trap_image_hashes']:
            channel_mappings[user_id][pair_name]['trap_image_hashes'].append(phash)
            save_mappings()
            await event.reply(f"🚫 Added block image hash for '{pair_name}': {phash}")
            logger.info(f"Added block image hash '{phash}' to pair '{pair_name}' for user {user_id}")
        else:
            await event.reply(f"⏸️ Image hash already exists in '{pair_name}'.")
    except Exception as e:
        logger.error(f"Error adding block image: {e}")
        await event.reply("❌ Failed to add block image.")

@client.on(events.NewMessage(pattern=r'/removeblockimage (\S+) (\S+)'))
async def remove_block_image(event: events.NewMessage.Event):
    """Remove an image’s perceptual hash from block list."""
    try:
        pair_name, phash = event.pattern_match.groups()
        user_id = str(event.sender_id)
        pair_name = pair_name.strip()
        phash = phash.strip()
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        if phash in channel_mappings[user_id][pair_name]['trap_image_hashes']:
            channel_mappings[user_id][pair_name]['trap_image_hashes'].remove(phash)
            save_mappings()
            await event.reply(f"🚫 Removed block image hash from '{pair_name}': {phash}")
            logger.info(f"Removed block image hash '{phash}' from pair '{pair_name}' for user {user_id}")
        else:
            await event.reply(f"❌ Image hash not found in '{pair_name}'.")
    except Exception as e:
        logger.error(f"Error removing block image: {e}")
        await event.reply("❌ Failed to remove block image.")

@client.on(events.NewMessage(pattern=r'/showblocks (\S+)'))
async def show_blocks(event: events.NewMessage.Event):
    """Show block filters for a pair."""
    try:
        pair_name = event.pattern_match.group(1).strip()
        user_id = str(event.sender_id)
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        mapping = channel_mappings[user_id][pair_name]
        blocks_info = [
            f"🚫 Block Filters for pair '{pair_name}':",
            f"Trap Phrases: {', '.join(mapping['trap_phrases']) or 'None'}",
            f"Trap Image Hashes: {', '.join(mapping['trap_image_hashes']) or 'None'}"
        ]
        await event.reply("\n".join(blocks_info))
        logger.info(f"Showed block filters for pair '{pair_name}' for user {user_id}")
    except Exception as e:
        logger.error(f"Error showing block filters: {e}")
        await event.reply("❌ Failed to show block filters.")

@client.on(events.NewMessage(pattern=r'/addhoneypot (\S+) (.+)'))
async def add_honeypot(event: events.NewMessage.Event):
    """Add a honeypot phrase."""
    try:
        pair_name, phrase = event.pattern_match.groups()
        user_id = str(event.sender_id)
        pair_name = pair_name.strip()
        phrase = phrase.strip()
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        if phrase not in channel_mappings[user_id][pair_name]['honeypot_phrases']:
            channel_mappings[user_id][pair_name]['honeypot_phrases'].append(phrase)
            save_mappings()
            await event.reply(f"🪤 Added honeypot phrase for '{pair_name}': {phrase}")
            logger.info(f"Added honeypot phrase '{phrase}' to pair '{pair_name}' for user {user_id}")
        else:
            await event.reply(f"⏸️ Honeypot phrase already exists in '{pair_name}'.")
    except Exception as e:
        logger.error(f"Error adding honeypot phrase: {e}")
        await event.reply("❌ Failed to add honeypot phrase.")

@client.on(events.NewMessage(pattern=r'/removehoneypot (\S+) (.+)'))
async def remove_honeypot(event: events.NewMessage.Event):
    """Remove a honeypot phrase."""
    try:
        pair_name, phrase = event.pattern_match.groups()
        user_id = str(event.sender_id)
        pair_name = pair_name.strip()
        phrase = phrase.strip()
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return
        if phrase in channel_mappings[user_id][pair_name]['honeypot_phrases']:
            channel_mappings[user_id][pair_name]['honeypot_phrases'].remove(phrase)
            save_mappings()
            await event.reply(f"🪤 Removed honeypot phrase from '{pair_name}': {phrase}")
            logger.info(f"Removed honeypot phrase '{phrase}' from pair '{pair_name}' for user {user_id}")
        else:
            await event.reply(f"❌ Honeypot phrase not found in '{pair_name}'.")
    except Exception as e:
        logger.error(f"Error removing honeypot phrase: {e}")
        await event.reply("❌ Failed to remove honeypot phrase.")

@client.on(events.NewMessage(pattern=r'/showhoneypots (\S+)'))
async def show_honeypots(event: events.NewMessage.Event):
    """Show honeypot phrases for a pair."""
    try:
        pair_name = event.pattern_match.group(1).strip()
        user_id = str(event.sender_id)
        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")