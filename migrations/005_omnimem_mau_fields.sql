-- Migration 005: OmniMEM MAU (Multimodal Atomic Unit) fields
--
-- Adds modality and raw_data_pointer columns to entities table,
-- formalizing the MAU tuple structure from OmniMEM (April 2026):
--
-- MAU = (S=summary, E=embedding, P=raw_data_pointer, τ=timestamp, M=modality, L=links)
--
-- Our existing entities table already has:
--   S → observations table (text summaries/facts)
--   E → Qdrant vector embeddings (via neural_memory_fabric)
--   τ → created_at timestamp
--   L → relations + memory_associations tables
--
-- This migration adds the missing fields:
--   P → raw_data_pointer (file path to heavy raw data: images, audio, video)
--   M → modality (text, image, audio, video, code, mixed, structured)

-- Add modality column: what type of content this entity represents
ALTER TABLE entities ADD COLUMN modality TEXT DEFAULT 'text';

-- Add raw_data_pointer: file path to heavy/binary content stored outside SQLite
ALTER TABLE entities ADD COLUMN raw_data_pointer TEXT;

-- Index on modality for filtered queries (e.g., "find all image memories")
CREATE INDEX IF NOT EXISTS idx_entities_modality ON entities(modality);
