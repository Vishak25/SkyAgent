interface Props {
  summary: string | null
}

export default function LLMSummary({ summary }: Props) {
  if (!summary) return null

  return (
    <div className="card llm-summary">
      <div className="llm-header">
        <h3 className="card-title">AI Assessment</h3>
      </div>
      <p className="llm-text">{summary}</p>
    </div>
  )
}
