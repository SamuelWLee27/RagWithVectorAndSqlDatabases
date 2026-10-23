"""Settings shared by the setup scripts and the app."""

# Weaviate vector data
COLLECTION = "ProductDocs"

# Must be the same model for storing chunks and embedding questions
EMBED_MODEL = "nomic-embed-text"