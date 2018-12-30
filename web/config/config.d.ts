import { LoggerOptions } from 'pino'
import { ConnectionOptions } from 'typeorm'

export interface Config {
  server: {
    port: number
    maxMemory: number
    killTimeout: number
  }

  database: {
    type: "sqlite"
    database: string
  }

  logging: LoggerOptions

}
