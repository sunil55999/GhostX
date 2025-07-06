import asyncio
import logging
import json
import re
import shutil
import random
import os
import signal
import sys
from datetime import datetime, timezone, timedelta
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from telethon import TelegramClient, events, errors
from telethon.tl.types import (
    MessageEntityBankCard, MessageEntityBlockquote, MessageEntityBold,
    MessageEntityBotCommand, MessageEntityCashtag, MessageEntityCode,
    MessageEntityCustomEmoji, MessageEntityEmail, MessageEntityHashtag,
    MessageEntityItalic, MessageEntityMention, MessageEntityMentionName,
    MessageEntityPhone, MessageEntityPre, MessageEntitySpoiler,
    MessageEntityStrike, MessageEntityTextUrl, MessageEntityUnderline,
    MessageEntityUnknown, MessageEntityUrl, MessageMediaPhoto, MessageMediaDocument
)
from PIL import Image
import io
from dotenv import load_dotenv
import pytz
from datetime import time

# Custom Modules
import reply_mapper
import trap_filter
import stealth_utils

# --- Configuration ---
load_dotenv()

# --- File Paths (must be defined before logging setup) ---
SESSION_FILE = "ghostcopy.session"
MAPPINGS_FILE = "channel_mappings.json"
LOG_FILE = "ghostcopybotpro.log"

# --- Logging Setup (must be before any logging calls) ---
def setup_logging(stealth=False):
    """Configure logging based on stealth mode."""
    global STEALTH_MODE
    STEALTH_MODE = stealth
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    log_level = logging.WARNING if stealth else logging.INFO
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # General log handler
    main_handler = logging.FileHandler(LOG_FILE)
    main_handler.setLevel(log_level)
    main_handler.setFormatter(log_format)
    
    # Failure log handler
    failure_handler = logging.FileHandler("copy_failures.log")
    failure_handler.setLevel(logging.ERROR)
    failure_handler.setFormatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)

    root_logger.setLevel(log_level)
    root_logger.addHandler(main_handler)
    root_logger.addHandler(failure_handler)
    root_logger.addHandler(console_handler)
    
    logging.getLogger('telethon').setLevel(logging.WARNING)
    global logger
    logger = logging.getLogger("GhostCopyBotPro")
    logger.info(f"Logging initialized. Stealth mode: {'ON' if stealth else 'OFF'}")

setup_logging()  # Initialize logging before any log calls
logger = logging.getLogger("GhostCopyBotPro")

API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')
# OWNER_ID = os.getenv('OWNER_ID')
# logger.info(f"Loaded OWNER_ID from .env: {OWNER_ID}")
# if not OWNER_ID:
#     logger.critical("OWNER_ID not found in environment. Check your .env file location and formatting.")
#     raise ValueError("OWNER_ID must be set in the .env file. It is required for all bot features.")
# OWNER_ID = int(OWNER_ID)

# Performance & Limits
MAX_RETRIES = 3
RETRY_DELAY = 5
MAX_QUEUE_SIZE = 200
MAX_MAPPING_HISTORY = 2000 # Kept in DB now, this is for local cache if needed
MAX_MESSAGE_LENGTH = 4096
NUM_WORKERS = 5
DEFAULT_DELAY_RANGE = [3.0, 7.0]
MAX_CHATS_PER_SESSION = 100

# --- Global State ---
channel_mappings: Dict[str, Dict[str, Dict[str, Any]]] = {}
message_queue: deque = deque(maxlen=MAX_QUEUE_SIZE)
queue_semaphore = asyncio.Semaphore(NUM_WORKERS)
is_connected = False
is_processing_enabled = True
pair_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
shutdown_event = asyncio.Event()
STEALTH_MODE = False # Toggle for logging
PAUSE_WINDOW = None # [time, time]

# --- Header/Footer Removal ---
HEADER_LIST = []  # Global headers to remove
FOOTER_LIST = []  # Global footers to remove

# Optionally, store per-pair headers/footers in channel_mappings

def remove_custom_header_footer(text: str, headers: list, footers: list) -> str:
    """
    Remove any header at the start and footer at the end of the message.
    Preserves spacing and formatting as much as possible.
    """
    cleaned = text
    for header in headers:
        if cleaned.startswith(header):
            cleaned = cleaned[len(header):].lstrip()
    for footer in footers:
        if cleaned.endswith(footer):
            cleaned = cleaned[:-len(footer)].rstrip()
    return cleaned

# --- Pause/Resume All Pairs ---
PAUSE_ALL = False
PAUSE_ALL_LOG = None

# --- Client Initialization ---
client = TelegramClient(
    SESSION_FILE,
    API_ID,
    API_HASH,
    **stealth_utils.get_random_device_info()
)

# --- Database & Mappings ---
async def initialize_databases():
    """Initialize all necessary databases."""
    await reply_mapper.init_db()

def save_mappings():
    """Save channel mappings to file."""
    try:
        with open(MAPPINGS_FILE, "w") as f:
            json.dump(channel_mappings, f, indent=2)
        if not STEALTH_MODE:
            logger.info("Channel mappings saved.")
    except Exception as e:
        logger.error(f"Error saving mappings: {e}")

def load_mappings():
    """Load channel mappings from file and set defaults."""
    global channel_mappings, PAUSE_WINDOW
    try:
        if os.path.exists(MAPPINGS_FILE):
            with open(MAPPINGS_FILE, "r") as f:
                data = json.load(f)
            
            # Load pause window settings first
            pause_window_str = data.pop('pause_window', None)
            if pause_window_str and len(pause_window_str) == 2:
                try:
                    PAUSE_WINDOW = [
                        time.fromisoformat(pause_window_str[0]),
                        time.fromisoformat(pause_window_str[1])
                    ]
                    logger.info(f"Loaded global pause window: {PAUSE_WINDOW[0].strftime('%H:%M')} - {PAUSE_WINDOW[1].strftime('%H:%M')} IST")
                except (ValueError, TypeError):
                    logger.error("Invalid pause window format in mappings file. Ignoring.")
                    PAUSE_WINDOW = None

            channel_mappings = data # The rest is channel mappings
            logger.info(f"Loaded {sum(len(v) for v in channel_mappings.values())} mappings.")
            for user_id, pairs in channel_mappings.items():
                pair_stats.setdefault(user_id, {})
                for pair_name, mapping in pairs.items():
                    mapping.setdefault('remove_mentions', True)
                    mapping.setdefault('trap_phrases', ['do_not_copy', 'trapword'])
                    mapping.setdefault('honeypot_phrases', [])
                    mapping.setdefault('trap_image_hashes', [])
                    mapping.setdefault('delay_range', DEFAULT_DELAY_RANGE)
                    mapping.setdefault('status', 'active')
                    pair_stats[user_id].setdefault(pair_name, {
                        'forwarded': 0, 'edited': 0, 'deleted': 0, 'blocked': 0, 'queued': 0, 'last_activity': None
                    })
        else:
            logger.info("No mappings file found. Starting fresh.")
    except json.JSONDecodeError:
        logger.error(f"Corrupted mappings file. Backing up and starting fresh.")
        shutil.move(MAPPINGS_FILE, MAPPINGS_FILE + ".bak")
        channel_mappings = {}
    except Exception as e:
        logger.error(f"Error loading mappings: {e}")

# --- Text & Entity Manipulation ---
def remove_mentions(text: str) -> str:
    """More robustly removes @mentions and t.me links."""
    # Regex to find @mentions, t.me links, and user profile links
    mention_pattern = re.compile(
        r'(?:\s|^)@\w+|https?://t\.me/(?:[a-zA-Z0-9_]+|joinchat/[a-zA-Z0-9_]+|\+[a-zA-Z0-9_]+)'
    )
    return mention_pattern.sub('', text).strip()

def adjust_entities(original_text: str, processed_text: str, entities: List[Any]) -> List[Any]:
    """
    Adjusts message entities after text modification.
    This remains a complex problem; a perfect solution is non-trivial.
    This version tries to be slightly more robust but can still fail on complex edits.
    """
    if not entities or not original_text:
        return []
    
    # Create a map of character indices from original to new text
    s = list(original_text)
    t = list(processed_text)
    
    matcher = SequenceMatcher(None, s, t)
    
    mapping = {}
    for i in range(len(s) + 1):
        mapping[i] = -1

    for block in matcher.get_matching_blocks():
        a, b, size = block
        for i in range(size):
            mapping[a + i] = b + i

    new_entities = []
    for entity in entities:
        start_offset = entity.offset
        end_offset = entity.offset + entity.length

        new_start = mapping.get(start_offset, -1)
        new_end = mapping.get(end_offset, -1)

        if new_start != -1 and new_end != -1:
            new_length = new_end - new_start
            if new_length > 0:
                try:
                    # Recreate the entity, preserving its type and any extra attributes
                    entity_args = {k: v for k, v in entity.to_dict().items() if k not in ['_', 'offset', 'length']}
                    new_entity = type(entity)(offset=new_start, length=new_length, **entity_args)
                    new_entities.append(new_entity)
                except Exception as e:
                    if not STEALTH_MODE:
                        logger.warning(f"Could not reconstruct entity: {e}")

    return new_entities

# --- Core Message Processing ---
async def process_and_copy_message(
    event: events.NewMessage.Event,
    mapping: Dict[str, Any],
    user_id: str,
    pair_name: str
):
    """The main logic for processing and copying a single message."""
    source_chat_id = event.chat_id
    source_msg_id = event.message.id
    dest_chat_id = int(mapping['destination'])

    for attempt in range(MAX_RETRIES):
        try:
            # --- Respect global pause ---
            if PAUSE_ALL:
                logger.info(f"[PAUSE_ALL] Skipping message for pair '{pair_name}' (paused globally)")
                return
            # 1. Trap Detection
            is_trap, reason = trap_filter.is_trap_candidate(
                event.message.raw_text,
                mapping.get('trap_phrases', []),
                mapping.get('honeypot_phrases', [])
            )
            if is_trap:
                if "Honeypot" not in reason:
                    await notify_trap(event, mapping, pair_name, reason)
                    pair_stats[user_id][pair_name]['blocked'] += 1
                return

            # 2. Text Processing
            processed_text = event.message.raw_text or ""
            original_entities = event.message.entities or []
            processed_entities = original_entities
            if processed_text and mapping.get('remove_mentions', True):
                processed_text = remove_mentions(processed_text)
                processed_entities = adjust_entities(event.message.raw_text, processed_text, original_entities)

            # --- Remove custom header/footer before sending ---
            processed_text = remove_custom_header_footer(processed_text, HEADER_LIST, FOOTER_LIST)

            # 3. Media Processing (Force Copy)
            processed_media = None
            if event.message.media:
                try:
                    if isinstance(event.message.media, MessageMediaPhoto):
                        image_bytes = await client.download_media(event.message, bytes)
                        if image_bytes:
                            processed_media, is_trap, reason = await stealth_utils.process_stealth_image(
                                image_bytes, mapping.get('trap_image_hashes', [])
                            )
                            if is_trap:
                                await notify_trap(event, mapping, pair_name, reason)
                                pair_stats[user_id][pair_name]['blocked'] += 1
                                return
                    else: # Handle all other media types (videos, documents, etc.)
                        # Download to a BytesIO object to re-upload
                        processed_media = io.BytesIO()
                        await client.download_media(event.message, processed_media)
                        processed_media.seek(0)
                        # Preserve original filename if possible
                        if hasattr(event.message.media.document, 'attributes'):
                            for attr in event.message.media.document.attributes:
                                if hasattr(attr, 'file_name'):
                                    processed_media.name = attr.file_name
                                    break
                except Exception as e:
                    logger.error(f"Failed to process media for pair '{pair_name}': {e}", exc_info=True)
                    return # Skip message if media fails

            # 4. Final Check
            if not processed_text.strip() and not processed_media:
                await notify_trap(event, mapping, pair_name, "Message empty after filtering")
                pair_stats[user_id][pair_name]['blocked'] += 1
                return

            # 5. Handle Replies
            reply_to_id = None
            if event.message.reply_to:
                source_reply_id = event.message.reply_to.reply_to_msg_id
                reply_to_id = await reply_mapper.get_dest_message_id(source_chat_id, source_reply_id, dest_chat_id)

            # 6. Apply Delay
            min_delay, max_delay = mapping.get('delay_range', DEFAULT_DELAY_RANGE)
            await asyncio.sleep(random.uniform(min_delay, max_delay))

            # 7. Send Message
            sent_message = await client.send_message(
                entity=dest_chat_id,
                message=processed_text,
                file=processed_media,
                reply_to=reply_to_id,
                silent=event.message.silent,
                formatting_entities=processed_entities
            )

            # 8. Store Mapping & Stats
            if sent_message:
                await reply_mapper.store_message_mapping(source_chat_id, source_msg_id, dest_chat_id, sent_message.id)
                pair_stats[user_id][pair_name]['forwarded'] += 1
                pair_stats[user_id][pair_name]['last_activity'] = datetime.now(timezone.utc).isoformat()
                if not STEALTH_MODE:
                    logger.info(f"Copied message for pair '{pair_name}'")
            
            return # Success, exit retry loop

        except (errors.FloodWaitError, errors.PeerFloodError) as e:
            wait_time = e.seconds if hasattr(e, 'seconds') else RETRY_DELAY * (2 ** attempt)
            logger.warning(f"Network flood error for '{pair_name}'. Attempt {attempt + 1}/{MAX_RETRIES}. Retrying in {wait_time}s.")
            await asyncio.sleep(wait_time + random.uniform(1, 3))
        except (errors.ChatWriteForbiddenError, errors.ChannelInvalidError) as e:
            logger.error(f"Cannot write to {mapping['destination']} for pair '{pair_name}'. Pausing. Error: {e}")
            mapping['status'] = 'paused'
            save_mappings()
            return
        except errors.MessageIdInvalidError:
            logger.warning(f"Invalid message ID for reply in '{pair_name}'. Sending without reply.")
            reply_to_id = None # Clear reply_to and retry
            continue # Go to next attempt without sleeping
        except Exception as e:
            logger.error(f"Unhandled error in process_and_copy_message for '{pair_name}' (Attempt {attempt + 1}): {e}", exc_info=True)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                logger.critical(f"Failed to copy message for '{pair_name}' after {MAX_RETRIES} attempts.")
                return

async def notify_trap(event, mapping, pair_name, reason):
    """Notify of trap detection (logging only, no owner message)."""
    msg_id = getattr(event.message, 'id', 'Unknown')
    logger.warning(f"Block detected in pair '{pair_name}' from '{mapping['source']}'. Reason: {reason}. Source Message ID: {msg_id}")

# --- Event Handlers ---
def is_in_pause_window() -> bool:
    """Check if the current time is within the defined pause window (IST)."""
    if not PAUSE_WINDOW:
        return False
    
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(ist).time()
    
    start, end = PAUSE_WINDOW
    
    # Handle overnight window (e.g., 22:00 to 06:00)
    if start > end:
        return now_ist >= start or now_ist < end
    else:
        return start <= now_ist < end

@client.on(events.NewMessage)
async def on_new_message(event: events.NewMessage.Event):
    """Main handler for incoming messages."""
    if not is_processing_enabled or not is_connected:
        return

    if is_in_pause_window():
        if not STEALTH_MODE:
            logger.info("Global pause window is active. Suppressing forwarding.")
        return
    
    source_chat_id = str(event.chat_id)
    
    destinations = []
    for user_id, pairs in channel_mappings.items():
        for pair_name, mapping in pairs.items():
            if mapping['status'] == 'active' and mapping['source'] == source_chat_id:
                destinations.append((event, mapping, user_id, pair_name))
    
    if not destinations:
        return

    random.shuffle(destinations)
    for dest_info in destinations:
        if len(message_queue) < MAX_QUEUE_SIZE:
            message_queue.append(dest_info)
            user_id, pair_name = dest_info[2], dest_info[3]
            pair_stats[user_id][pair_name]['queued'] += 1
        else:
            logger.warning(f"Queue is full. Dropping message for pair '{dest_info[3]}'.")

@client.on(events.MessageEdited)
async def on_message_edited(event: events.MessageEdited.Event):
    """Handle message edits to keep copies in sync."""
    if not is_processing_enabled or not is_connected:
        return

    if is_in_pause_window():
        if not STEALTH_MODE:
            logger.info("Global pause window is active. Suppressing message edit.")
        return

    source_chat_id = str(event.chat_id)
    source_msg_id = event.message.id

    dest_messages = await reply_mapper.get_all_dest_messages(source_chat_id, source_msg_id)
    if not dest_messages:
        return

    for dest_chat_id, dest_msg_id in dest_messages:
        found = False
        for user_id, pairs in channel_mappings.items():
            for pair_name, mapping in pairs.items():
                if mapping['source'] == source_chat_id and mapping['destination'] == str(dest_chat_id):
                    found = True
                    try:
                        # Re-run filters on the edited content
                        is_trap, reason = trap_filter.is_trap_candidate(
                            event.message.raw_text,
                            mapping.get('trap_phrases', []),
                            mapping.get('honeypot_phrases', [])
                        )
                        if is_trap:
                            await client.delete_messages(dest_chat_id, [dest_msg_id])
                            if "Honeypot" not in reason:
                                await notify_trap(event, mapping, pair_name, f"Message deleted on edit: {reason}")
                            continue
                        cleaned_text = event.message.raw_text or ""
                        original_entities = event.message.entities or []
                        if mapping.get('remove_mentions', True):
                            cleaned_text = remove_mentions(cleaned_text)
                        cleaned_text = remove_custom_header_footer(cleaned_text, HEADER_LIST, FOOTER_LIST)
                        cleaned_entities = adjust_entities(event.message.raw_text, cleaned_text, original_entities)
                        if not cleaned_text.strip() and not event.message.media:
                            await client.delete_messages(dest_chat_id, [dest_msg_id])
                            await notify_trap(event, mapping, pair_name, "Message deleted on edit: empty after filtering")
                            continue
                        await asyncio.sleep(random.uniform(2.0, 5.0))
                        try:
                            await client.edit_message(
                                entity=dest_chat_id,
                                message=dest_msg_id,
                                text=cleaned_text,
                                entities=cleaned_entities
                            )
                            pair_stats[user_id][pair_name]['edited'] += 1
                            if not STEALTH_MODE:
                                logger.info(f"Edited message in pair '{pair_name}'")
                        except errors.MessageNotModifiedError:
                            logger.info(f"Message not modified for pair '{pair_name}' (no changes)")
                        except errors.MessageIdInvalidError:
                            logger.warning(f"Invalid message ID for edit in pair '{pair_name}'")
                        except errors.FloodWaitError as e:
                            logger.warning(f"Flood wait error while editing message in pair '{pair_name}': {e}")
                        except Exception as e:
                            logger.error(f"Error editing message for pair '{pair_name}': {e}", exc_info=True)
                    except Exception as e:
                        logger.error(f"Error in edit sync for pair '{pair_name}': {e}", exc_info=True)
        if not found:
            logger.warning(f"No mapping found for edited message {source_msg_id} in chat {source_chat_id} -> {dest_chat_id}")

@client.on(events.MessageDeleted)
async def on_message_deleted(event: events.MessageDeleted.Event):
    """Handle message deletions to keep copies in sync."""
    if not is_processing_enabled or not is_connected or not event.deleted_ids:
        return

    if is_in_pause_window():
        if not STEALTH_MODE:
            logger.info("Global pause window is active. Suppressing message deletion.")
        return

    source_chat_id = str(event.chat_id)
    
    for deleted_id in event.deleted_ids:
        dest_messages = await reply_mapper.get_all_dest_messages(source_chat_id, deleted_id)
        if not dest_messages:
            continue

        for dest_chat_id, dest_msg_id in dest_messages:
            try:
                await client.delete_messages(dest_chat_id, [dest_msg_id])
                
                # Correctly find the user and pair to update stats
                user_to_update = None
                pair_to_update = None
                for user_id, pairs in channel_mappings.items():
                    for pair_name, mapping in pairs.items():
                        if mapping['source'] == source_chat_id and mapping['destination'] == str(dest_chat_id):
                            user_to_update = user_id
                            pair_to_update = pair_name
                            break
                    if user_to_update:
                        break
                
                if user_to_update and pair_to_update:
                    pair_stats[user_to_update][pair_to_update]['deleted'] += 1

                if not STEALTH_MODE:
                    logger.info(f"Deleted message {dest_msg_id} from chat {dest_chat_id}")
            except Exception as e:
                logger.error(f"Error deleting message {dest_msg_id}: {e}")

# --- Bot Commands ---
@client.on(events.NewMessage(pattern='/traptest'))
async def traptest_command(event: events.NewMessage.Event):
    """Command to test why a message would be blocked."""
    if not event.reply_to_msg_id:
        await event.reply("Reply to a message to test it.")
        return
        
    replied_msg = await event.get_reply_message()
    
    # Test text
    is_trap, reason = trap_filter.is_trap_candidate(
        replied_msg.raw_text,
        ['trapword', 'blockthis'], # Example trap words
        ['do_not_copy'] # Example honeypot
    )
    text_report = f"Text Analysis:\n- Is Trap: {is_trap}\n- Reason: {reason}\n"
    
    # Test image
    image_report = "Image Analysis:\n- No image found.\n"
    if isinstance(replied_msg.media, MessageMediaPhoto):
        image_bytes = await client.download_media(replied_msg, bytes)
        if image_bytes:
            _, is_trap, reason = await stealth_utils.process_stealth_image(
                image_bytes, ['some_fake_hash'] # Example hash
            )
            image_report = f"Image Analysis:\n- Is Trap: {is_trap}\n- Reason: {reason or 'Clean'}\n"

    await event.reply(f"🕵️ Trap Test Report:\n\n{text_report}\n{image_report}")

@client.on(events.NewMessage(pattern=r'/addblockimage (\S+)'))
async def add_block_image_command(event: events.NewMessage.Event):
    """Adds an image's perceptual hashes to the block list for a pair."""
    try:
        pair_name = event.pattern_match.group(1).strip()
        user_id = str(event.sender_id)

        if user_id not in channel_mappings or pair_name not in channel_mappings[user_id]:
            await event.reply("❌ Pair not found.")
            return

        if not event.message.reply_to:
            await event.reply("🖼️ Please reply to a photo to block it.")
            return
            
        replied_msg = await event.get_reply_message()
        if not isinstance(replied_msg.media, MessageMediaPhoto):
            await event.reply("🖼️ The replied message is not a photo.")
            return

        image_bytes = await client.download_media(replied_msg, bytes)
        if not image_bytes:
            await event.reply("❌ Failed to download photo.")
            return

        image = Image.open(io.BytesIO(image_bytes))
        hashes = stealth_utils.get_image_hashes(image)
        
        if not hashes:
            await event.reply("❌ Could not generate hashes for this image.")
            return

        trap_list = channel_mappings[user_id][pair_name].setdefault('trap_image_hashes', [])
        new_hashes = []
        for hash_type, hash_value in hashes.items():
            if hash_value not in trap_list:
                trap_list.append(hash_value)
                new_hashes.append(f"{hash_type}: {hash_value}")

        if new_hashes:
            save_mappings()
            await event.reply(f"🚫 Added {len(new_hashes)} new image hashes to block list for '{pair_name}':\n" + "\n".join(new_hashes))
            logger.info(f"Added {len(new_hashes)} image hashes to pair '{pair_name}' for user {user_id}")
        else:
            await event.reply("✅ All hashes for this image are already in the block list.")

    except Exception as e:
        logger.error(f"Error in /addblockimage: {e}")
        await event.reply("❌ An error occurred while adding the block image.")

@client.on(events.NewMessage(pattern='/flushcache'))
async def flush_cache_command(event: events.NewMessage.Event):
    await reply_mapper.flush_mappings()
    # Also clear any file-based caches if they exist
    await event.reply("✅ All reply mappings have been flushed.")

@client.on(events.NewMessage(pattern=r'/setpair (\S+) (\S+) (\S+)'))
async def set_pair_command(event: events.NewMessage.Event):
    """Sets up a new source-destination pair."""
    try:
        pair_name, source, dest = event.pattern_match.groups()
        user_id = str(event.sender_id)
        
        channel_mappings.setdefault(user_id, {})
        pair_stats.setdefault(user_id, {})

        channel_mappings[user_id][pair_name] = {
            'source': source,
            'destination': dest,
            'status': 'active',
            'remove_mentions': True,
            'trap_phrases': ['do_not_copy', 'trapword'],
            'honeypot_phrases': [],
            'trap_image_hashes': [],
            'delay_range': DEFAULT_DELAY_RANGE,
        }
        pair_stats[user_id][pair_name] = {'forwarded': 0, 'edited': 0, 'deleted': 0, 'blocked': 0, 'queued': 0, 'last_activity': None}
        
        save_mappings()
        await event.reply(f"✅ Pair '{pair_name}' created: {source} -> {dest}")
    except Exception as e:
        await event.reply(f"❌ Error setting pair: {e}")

@client.on(events.NewMessage(pattern=r'/delpair (\S+)'))
async def del_pair_command(event: events.NewMessage.Event):
    """Deletes a forwarding pair."""
    try:
        pair_name = event.pattern_match.group(1)
        user_id = str(event.sender_id)
        if channel_mappings.get(user_id, {}).pop(pair_name, None):
            pair_stats.get(user_id, {}).pop(pair_name, None)
            save_mappings()
            await event.reply(f"✅ Pair '{pair_name}' deleted.")
        else:
            await event.reply("❌ Pair not found.")
    except (IndexError, Exception) as e:
        await event.reply(f"❌ Error deleting pair. Usage: /delpair <pair_name>. Error: {e}")

@client.on(events.NewMessage(pattern='/listpairs'))
async def list_pairs_command(event: events.NewMessage.Event):
    user_id = str(event.sender_id)
    if not channel_mappings.get(user_id):
        await event.reply("No pairs configured.")
        return
    
    reply = "📜 Configured Pairs:\n\n"
    for name, m in channel_mappings[user_id].items():
        reply += f"**{name}**: {m['source']} -> {m['destination']} (Status: {m['status']})\n"
    
    await event.reply(reply)

@client.on(events.NewMessage(pattern=r'/stealthmode (\S+)'))
async def stealth_mode_command(event: events.NewMessage.Event):
    try:
        mode = event.pattern_match.group(1).lower()
        if mode == "on":
            setup_logging(stealth=True)
            await event.reply("Stealth mode **ON**. Logging is now minimal.")
        elif mode == "off":
            setup_logging(stealth=False)
            await event.reply("Stealth mode **OFF**. Verbose logging is enabled.")
        else:
            await event.reply("Usage: /stealthmode <on|off>")
    except IndexError:
        await event.reply(f"Stealth mode is currently **{'ON' if STEALTH_MODE else 'OFF'}**.\nUsage: /stealthmode <on|off>")

@client.on(events.NewMessage(pattern='/report'))
async def report_command(event: events.NewMessage.Event):
    user_id = str(event.sender_id)
    if not pair_stats.get(user_id):
        await event.reply("No stats to report.")
        return

    report = "📊 **Bot Activity Report**\n\n"
    for name, stats in pair_stats[user_id].items():
        report += (
            f"**Pair: {name}**\n"
            f"  - Forwarded: {stats['forwarded']}\n"
            f"  - Edited: {stats['edited']}\n"
            f"  - Deleted: {stats['deleted']}\n"
            f"  - Blocked: {stats['blocked']}\n"
            f"  - Queued: {len(message_queue)}/{MAX_QUEUE_SIZE}\n"
            f"  - Last Activity: {stats.get('last_activity', 'N/A')}\n\n"
        )
    await event.reply(report)

@client.on(events.NewMessage(pattern=r'/setpausewindow (\S+) (\S+)'))
async def set_pause_window_command(event: events.NewMessage.Event):
    global PAUSE_WINDOW
    try:
        start_str, end_str = event.pattern_match.groups()
        start_time = time.fromisoformat(start_str)
        end_time = time.fromisoformat(end_str)
        
        PAUSE_WINDOW = [start_time, end_time]
        
        # Save it to the mappings file
        mappings_data = channel_mappings
        mappings_data['pause_window'] = [start_time.isoformat(), end_time.isoformat()]
        with open(MAPPINGS_FILE, "w") as f:
            json.dump(mappings_data, f, indent=2)

        await event.reply(f"✅ Global pause window set from {start_str} to {end_str} IST.")
        logger.info(f"Global pause window set: {start_str} - {end_str} IST")
    except (ValueError, TypeError):
        await event.reply("❌ Invalid time format. Use HH:MM (e.g., /setpausewindow 23:00 07:00).")
    except Exception as e:
        await event.reply(f"❌ An error occurred: {e}")

@client.on(events.NewMessage(pattern='/clearpausewindow'))
async def clear_pause_window_command(event: events.NewMessage.Event):
    global PAUSE_WINDOW
    PAUSE_WINDOW = None
    
    # Correctly load, modify, and save the entire mappings file
    try:
        with open(MAPPINGS_FILE, "r") as f:
            mappings_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        mappings_data = {} # Start fresh if file is missing/corrupt

    if 'pause_window' in mappings_data:
        del mappings_data['pause_window']
    
    with open(MAPPINGS_FILE, "w") as f:
        json.dump(mappings_data, f, indent=2)
        
    await event.reply("✅ Global pause window cleared.")
    logger.info("Global pause window cleared.")

@client.on(events.NewMessage(pattern='/testcopy'))
async def test_copy_command(event: events.NewMessage.Event):
    """Simulates the full copy process for a replied-to message."""
    if not event.reply_to_msg_id:
        await event.reply("Reply to a message to test the copy process.")
        return
        
    replied_msg = await event.get_reply_message()
    
    report = "🧪 **Copy Simulation Report**\n\n"
    
    # 1. Text Processing
    original_text = replied_msg.raw_text or ""
    processed_text = remove_mentions(original_text)
    report += f"**Original Text:**\n`{original_text}`\n\n"
    report += f"**Processed Text (Mentions Removed):**\n`{processed_text}`\n\n"
    
    # 2. Entity Adjustment
    original_entities = replied_msg.entities or []
    new_entities = adjust_entities(original_text, processed_text, original_entities)
    report += f"**Entities:**\n- Original Count: {len(original_entities)}\n- Adjusted Count: {len(new_entities)}\n\n"

    # 3. Media Processing
    media_report = "**Media Processing:**\n- No media found.\n"
    if replied_msg.media:
        if isinstance(replied_msg.media, MessageMediaPhoto):
            try:
                image_bytes = await client.download_media(replied_msg, bytes)
                _, is_trap, reason = await stealth_utils.process_stealth_image(image_bytes, [])
                media_report = (
                    f"**Media Processing (Image):**\n"
                    f"- Is Trap: {is_trap}\n"
                    f"- Reason: {reason or 'Clean'}\n"
                    f"- Obfuscation would be applied.\n"
                )
            except Exception as e:
                media_report = f"**Media Processing (Image):**\n- Error: {e}\n"
        else:
            media_report = f"**Media Processing (Other):**\n- Type: {type(replied_msg.media).__name__}\n- Would be downloaded and re-uploaded.\n"
    
    report += media_report
    
    await event.reply(report, parse_mode='markdown')

@client.on(events.NewMessage(pattern='/help'))
async def help_command(event: events.NewMessage.Event):
    """Lists all available bot commands and their usage."""
    help_text = (
        "📖 **GhostCopyBotPro Commands**\n\n"
        "/help - Show this help message\n"
        "/traptest - Test why a message would be blocked (reply to a message)\n"
        "/addblockimage <pair_name> - Add a photo's hashes to block list (reply to a photo)\n"
        "/flushcache - Flush all reply mappings\n"
        "/setpair <pair_name> <source> <destination> - Create a new forwarding pair\n"
        "/delpair <pair_name> - Delete a forwarding pair\n"
        "/listpairs - List all configured pairs\n"
        "/stealthmode <on|off> - Toggle stealth mode (logging)\n"
        "/report - Show a statistical report of bot activity\n"
        "/setpausewindow <start> <end> - Set the global pause window (HH:MM HH:MM)\n"
        "/clearpausewindow - Clear the global pause window\n"
        "/testcopy - Simulate the copy process for a replied-to message\n"
        "\n"
        "*Some commands may be restricted if OWNER_ID is not set.*"
    )
    await event.reply(help_text, parse_mode='markdown')

@client.on(events.NewMessage(pattern=r'/setheader (.+)'))
async def set_header_command(event: events.NewMessage.Event):
    """Set a custom header to remove from all messages."""
    global HEADER_LIST
    header = event.pattern_match.group(1).strip()
    if header:
        if header not in HEADER_LIST:
            HEADER_LIST.append(header)
            await event.reply(f"✅ Header added: `{header}`")
        else:
            await event.reply(f"ℹ️ Header already exists: `{header}`")
    else:
        await event.reply("❌ Usage: /setheader <header text>")

@client.on(events.NewMessage(pattern=r'/setfooter (.+)'))
async def set_footer_command(event: events.NewMessage.Event):
    """Set a custom footer to remove from all messages."""
    global FOOTER_LIST
    footer = event.pattern_match.group(1).strip()
    if footer:
        if footer not in FOOTER_LIST:
            FOOTER_LIST.append(footer)
            await event.reply(f"✅ Footer added: `{footer}`")
        else:
            await event.reply(f"ℹ️ Footer already exists: `{footer}`")
    else:
        await event.reply("❌ Usage: /setfooter <footer text>")

@client.on(events.NewMessage(pattern='/clearhf'))
async def clear_hf_command(event: events.NewMessage.Event):
    """Clear all custom headers and footers."""
    global HEADER_LIST, FOOTER_LIST
    HEADER_LIST.clear()
    FOOTER_LIST.clear()
    await event.reply("✅ All headers and footers have been cleared.")

@client.on(events.NewMessage(pattern='/pauseall'))
async def pause_all_command(event: events.NewMessage.Event):
    """Pause message copying globally."""
    global PAUSE_ALL, PAUSE_ALL_LOG
    PAUSE_ALL = True
    PAUSE_ALL_LOG = datetime.now(timezone.utc).isoformat()
    await event.reply("⏸️ All message copying is now globally paused.")
    logger.info("Global pause activated by command.")

@client.on(events.NewMessage(pattern='/resumeall'))
async def resume_all_command(event: events.NewMessage.Event):
    """Resume all forwarding globally."""
    global PAUSE_ALL, PAUSE_ALL_LOG
    PAUSE_ALL = False
    PAUSE_ALL_LOG = None
    await event.reply("▶️ All message copying has been resumed.")
    logger.info("Global pause deactivated by command.")

# --- Main Application Logic ---
async def queue_processor():
    """The main worker loop that processes messages from the queue."""
    while not shutdown_event.is_set():
        try:
            if message_queue:
                event, mapping, user_id, pair_name = message_queue.popleft()
                async with queue_semaphore:
                    await process_and_copy_message(event, mapping, user_id, pair_name)
            else:
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Critical error in queue_processor: {e}", exc_info=True)
            await asyncio.sleep(1)

async def periodic_task_runner():
    """Runs periodic maintenance tasks."""
    while not shutdown_event.is_set():
        await asyncio.sleep(3600) # Run every hour
        try:
            await reply_mapper.cleanup_old_mappings()
        except Exception as e:
            logger.error(f"Error during periodic cleanup: {e}")

def shutdown_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    logger.warning("Shutdown signal received. Shutting down gracefully...")
    shutdown_event.set()

async def main():
    """Main entry point for the bot."""
    global is_connected, OWNER_ID
    
    setup_logging()
    await initialize_databases()
    load_mappings()
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Start worker tasks
    worker_tasks = [asyncio.create_task(queue_processor()) for _ in range(NUM_WORKERS)]
    periodic_task = asyncio.create_task(periodic_task_runner())
    
    logger.info("GhostCopyBotPro is starting...")
    await client.start()
    is_connected = True
    logger.info("Client connected successfully.")
    
    await shutdown_event.wait()
    
    # Graceful shutdown
    is_connected = False
    logger.info("Stopping worker tasks...")
    for task in worker_tasks:
        task.cancel()
    periodic_task.cancel()
    await asyncio.gather(*worker_tasks, periodic_task, return_exceptions=True)
    
    await client.disconnect()
    logger.info("Client disconnected. Shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (ValueError, TypeError) as e:
        # Catches the API_ID/HASH errors before the loop starts
        logger.critical(f"Configuration Error: {e}")