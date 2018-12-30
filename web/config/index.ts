import { Config } from './config'
import { config as installConfig } from 'dotenv'

installConfig()


export const config: Config = {
  server: {
    port: Number(process.env.PORT || 80),
    maxMemory: 512,
    killTimeout: 3000,
  },

  logging: {
    level: 'info',
    name: 'diploma-thesis-web-logger',
  },

  database: {
    database: './db/db.labels.data',
    type: 'sqlite',
  },

}
