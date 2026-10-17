"""nomic's task prefixes -- the adapter @sec-rag-foundations introduced."""

from langchain_openai import OpenAIEmbeddings


class NomicEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts, *args, **kwargs):
        return super().embed_documents([f"search_document: {t}" for t in texts], *args, **kwargs)

    def embed_query(self, text):
        # super().embed_documents, NOT embed_query: the base routes embed_query
        # through embed_documents, which would stack both prefixes.
        return super().embed_documents([f"search_query: {text}"])[0]


def make_embeddings(lm_base: str, model: str) -> NomicEmbeddings:
    return NomicEmbeddings(base_url=lm_base, api_key="lm-studio", model=model,
                           check_embedding_ctx_length=False, timeout=120)
