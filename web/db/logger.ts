import { Logger, QueryRunner } from 'typeorm'
import { log } from '../logger'

export class DbLogger implements Logger {
  /**
   * Logs query and parameters used in it.
   */
  logQuery(query: string, parameters?: any[], _queryRunner?: QueryRunner): any {
    log.debug({ query, parameters }, 'Executing TypeORM query...')
  }

  /**
   * Logs query that is failed.
   */
  logQueryError(error: string, query: string, parameters?: any[], _queryRunner?: QueryRunner): any {
    log.error({ error, query, parameters }, 'Error happened while executing TypeORM query...')
  }

  /**
   * Logs query that is slow.
   */
  logQuerySlow(time: number, query: string, parameters?: any[], _queryRunner?: QueryRunner): any {
    log.warn({ time, query, parameters }, 'This query is really slow...')
  }

  /**
   * Logs events from the schema build process.
   */
  logSchemaBuild(message: string, _queryRunner?: QueryRunner): any {
    log.debug({ message }, 'Running schema build process...')
  }

  /**
   * Logs events from the migrations run process.
   */
  logMigration(message: string, _queryRunner?: QueryRunner): any {
    log.debug({ message }, 'Running db migrations process...')
  }

  /**
   * Perform logging using given logger, or by default to the console.
   * Log has its own level and message.
   */
  log(level: 'log' | 'info' | 'warn', message: any, _queryRunner?: QueryRunner): any {
    log.debug({ level, message }, 'Logging TypeORM')
  }
}
