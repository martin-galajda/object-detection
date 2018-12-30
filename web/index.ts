import { api, Api } from './api'
import { log } from './logger'

api.start().then(() => {
  log.info('Api successfully started')
}).catch(Api.fatal)
