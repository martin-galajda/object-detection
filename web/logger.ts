import * as pino from 'pino'
import { config } from './config'

export const log = pino({
  ...config.logging,
  serializers: pino.stdSerializers,
})
