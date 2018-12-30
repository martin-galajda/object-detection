import { createConnection, Connection } from 'typeorm'
import { config } from '../config'
import { values } from 'ramda'
import * as models from '../models'
import { DbLogger } from './logger'

class Database {
  conn!: Connection

  protected _startDbPromise?: Promise<Connection>

  async start() {
    if (!this._startDbPromise) {
      this._startDbPromise = this._startDb()
    }

    return this._startDbPromise
  }

  async stop() {
    if (this.conn) {
      await this.conn.close()
    }
  }

  async _startDb() {
    this.conn = await createConnection({
      ...config.database,
      entities: values(models),
      logger: new DbLogger(),
      // maxQueryExecutionTime: 2,
    })

    return this.conn
  }
}

export const database = new Database()
