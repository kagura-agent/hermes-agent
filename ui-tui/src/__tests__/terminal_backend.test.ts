import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '../types.js'

describe('SessionInfo terminal_backend', () => {
  it('accepts terminal_backend field', () => {
    const info: SessionInfo = {
      model: 'test',
      skills: {},
      tools: {},
      terminal_backend: 'local',
    }
    expect(info.terminal_backend).toBe('local')
  })

  it('terminal_backend is optional', () => {
    const info: SessionInfo = {
      model: 'test',
      skills: {},
      tools: {},
    }
    expect(info.terminal_backend).toBeUndefined()
  })
})
