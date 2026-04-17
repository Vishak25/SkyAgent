import { useState, FormEvent } from 'react'

interface Props {
  onSearch: (flightNumber: string) => void
  loading: boolean
}

export default function FlightSearch({ onSearch, loading }: Props) {
  const [value, setValue] = useState('')

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    const trimmed = value.trim().toUpperCase()
    if (trimmed) onSearch(trimmed)
  }

  return (
    <form className="search-form" onSubmit={handleSubmit}>
      <div className="search-box">
        <span className="search-icon">✈</span>
        <input
          className="search-input"
          type="text"
          placeholder="Enter flight number (e.g. UAL123)"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          disabled={loading}
          spellCheck={false}
        />
        <button className="search-btn" type="submit" disabled={loading || !value.trim()}>
          {loading ? <span className="spinner" /> : 'Analyze'}
        </button>
      </div>
      <p className="search-hint">Powered by ST-GNN delay prediction + Qwen2.5-7B on GMU Hopper</p>
    </form>
  )
}
