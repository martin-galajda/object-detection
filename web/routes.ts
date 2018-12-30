import * as KoaRouter from 'koa-router'
import { Context } from 'koa'
import { LabelGroups } from './models'
export const router = new KoaRouter()

router.get('/', async (ctx: Context) => {
  const labelGroups = await LabelGroups.find()
  ctx.response.body = {
    groups: labelGroups,
  }
})
