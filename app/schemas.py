from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    id: str
    text: str
    source_document: str
    chunk_index: int
    chunk_total: int
    start_page: int = 0
    end_page: int = 0
    section: str = ""
    section_path: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)

    def embedding_input(self) -> str:
        parts: list[str] = []
        if self.source_document:
            parts.append(f"Document: {self.source_document}")
        if self.section_path:
            parts.append(f"Section: {self.section_path}")
        elif self.section:
            parts.append(f"Section: {self.section}")
        header = "\n".join(parts)
        if header:
            return f"{header}\n{self.text}"
        return self.text


class IngestResponse(BaseModel):
    document: str
    chunks_indexed: int
    message: str = "ok"


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceCitation(BaseModel):
    source_document: str
    section_path: str
    chunk_index: int
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    vertex: str
    reranker: str
