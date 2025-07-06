import aiosqlite
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

DB_PATH = "./reply_map.db"
MAX_MAPPING_AGE_DAYS = 7

async def init_db():
    """Initialize the reply mapping database."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS reply_map (
                source_chat_id INTEGER,
                source_msg_id INTEGER,
                dest_chat_id INTEGER,
                dest_msg_id INTEGER,
                timestamp TEXT,
                PRIMARY KEY (source_chat_id, source_msg_id, dest_chat_id)
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_source_message ON reply_map (source_chat_id, source_msg_id);
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON reply_map (timestamp);
        """)
        await db.commit()
    logger.info("Reply mapping database initialized.")

async def store_message_mapping(source_chat_id: int, source_msg_id: int, dest_chat_id: int, dest_msg_id: int):
    """Store a mapping from a source message to a destination message."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT OR REPLACE INTO reply_map (source_chat_id, source_msg_id, dest_chat_id, dest_msg_id, timestamp) VALUES (?, ?, ?, ?, ?)",
                (source_chat_id, source_msg_id, dest_chat_id, dest_msg_id, datetime.utcnow().isoformat())
            )
            await db.commit()
            logger.debug(f"Stored reply map: {source_chat_id}:{source_msg_id} -> {dest_chat_id}:{dest_msg_id}")
    except Exception as e:
        logger.error(f"Error storing message mapping: {e}")

async def get_dest_message_id(source_chat_id: int, source_reply_to_id: int, dest_chat_id: int) -> int | None:
    """Get the destination message ID that a source reply corresponds to."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                "SELECT dest_msg_id FROM reply_map WHERE source_chat_id = ? AND source_msg_id = ? AND dest_chat_id = ?",
                (source_chat_id, source_reply_to_id, dest_chat_id)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    logger.debug(f"Found reply map: {source_chat_id}:{source_reply_to_id} -> {dest_chat_id}:{row[0]}")
                    return row[0]
                else:
                    logger.debug(f"No reply map found for {source_chat_id}:{source_reply_to_id} in {dest_chat_id}")
                    return None
    except Exception as e:
        logger.error(f"Error retrieving destination message ID: {e}")
        return None

async def cleanup_old_mappings():
    """Remove old message mappings from the database."""
    try:
        cutoff_date = (datetime.utcnow() - timedelta(days=MAX_MAPPING_AGE_DAYS)).isoformat()
        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute("DELETE FROM reply_map WHERE timestamp < ?", (cutoff_date,))
            await db.commit()
            logger.info(f"Cleaned up {cursor.rowcount} old reply mappings.")
    except Exception as e:
        logger.error(f"Error cleaning up old mappings: {e}")

async def get_all_dest_messages(source_chat_id: int, source_msg_id: int) -> list[tuple[int, int]]:
    """Get all destination message IDs mapped from a single source message."""
    dest_messages = []
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                "SELECT dest_chat_id, dest_msg_id FROM reply_map WHERE source_chat_id = ? AND source_msg_id = ?",
                (source_chat_id, source_msg_id)
            ) as cursor:
                async for row in cursor:
                    dest_messages.append((row[0], row[1]))
        return dest_messages
    except Exception as e:
        logger.error(f"Error retrieving all destination messages: {e}")
        return []

async def flush_mappings():
    """Flush all mappings from the database."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute("DELETE FROM reply_map")
            await db.commit()
            logger.info(f"Flushed {cursor.rowcount} reply mappings from the database.")
    except Exception as e:
        logger.error(f"Error flushing mappings: {e}")