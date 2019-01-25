import aiosqlite
import asyncio
import io
import numpy as np

DB_FILE_PATH = './data/openimages/out/db.data'


async def update_data_column():

  try:
    async with aiosqlite.connect(DB_FILE_PATH, timeout = 1000.0, detect_types=aiosqlite.PARSE_DECLTYPES) as db:
      try:
        await db.executescript("""
          ALTER table images
          ADD COLUMN image_data BLOB;
        """)
        await db.commit()
      except Exception as e:
        print("Exception happened " + str(e))

  except Exception as e:
    print("Exception happened " + str(e))

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(update_data_column())

    print("Successfully updated data column!")