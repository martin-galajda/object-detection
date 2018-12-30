import { Server } from 'http'
import * as Koa from 'koa'
import * as koaCompress from 'koa-compress'
import * as koaBody from 'koa-body'
import * as koaCors from 'koa-cors'
import { config } from './config'
import { database } from './db'
import { log } from './logger'
import { router } from './routes'
// import { errorMiddleware } from 'app/middleware/error'

export class Api {
  static fatal(err: Error) {
    // Remove termination listener
    process.removeAllListeners('uncaughtException')
    process.removeAllListeners('unhandledRejection')

    log.fatal({ err }, 'Fatal error occurred. Exiting the app.')
  }

  server?: Server
  koa: Koa

  constructor() {
    this.koa = new Koa()
    this.attachMiddlewares()

    this.stop = this.stop.bind(this)
    this.start = this.start.bind(this)
  }

  attachMiddlewares() {
    this.koa.use(koaCompress())
    this.koa.use(koaCors({ origin: '*' }))
    // this.koa.use(errorMiddleware)
    this.koa.use(koaBody())
    this.koa.use(router.routes())
  }

  async start() {
    // Handle unexpected termination
    process.once('uncaughtException', Api.fatal)
    process.once('unhandledRejection', Api.fatal)

    // Handle expected termination
    process.once('SIGINT', async () => this.stop())
    process.once('SIGTERM', async () => this.stop())

    // Start database
    log.info('Starting database ...')
    await database.start()

    // Start server
    log.info('Starting server ...')
    await new Promise(done => {
      this.server = this.koa.listen(config.server.port, done)
    })
    log.info(`==> 🔥 Server is listening on port ${config.server.port}.`)
  }

  async stop() {
    // Remove listeners
    process.removeAllListeners('SIGINT')
    process.removeAllListeners('SIGTERM')

    // Check server is initialized
    const server = this.server
    if (server === undefined) {
      log.warn('Server not initialized yet.')
      return
    }

    // Close database connection
    log.info('Closing database connections.')
    await database.stop()

    // Stop server
    await new Promise(resolve => server.close(resolve))
    log.info('Server stopped.')
  }
}

export const api = new Api()
