import { Entity, Column, PrimaryGeneratedColumn, BaseEntity, CreateDateColumn } from 'typeorm'

@Entity('label_groups')
export class LabelGroups extends BaseEntity {
  @PrimaryGeneratedColumn()
  id: number

  @Column()
  labels_in_group: string

  @Column({ type: 'timestamp with time zone' })
  @CreateDateColumn()
  created_at: string
}
