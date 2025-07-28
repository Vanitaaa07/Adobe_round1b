import json
import os
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from summa import summarizer, keywords
from rake_nltk import Rake
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from collections import defaultdict, Counter
import numpy as np
import re

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm model for spaCy...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Download NLTK data
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")

stop_words = set(stopwords.words("english"))

class DocumentAnalystEnhanced:
    def __init__(self, persona_or_path, job_to_be_done):
        if isinstance(persona_or_path, dict):
            self.persona = persona_or_path
        else:
            self.persona = self._load_persona(persona_or_path)
        self.job_to_be_done = job_to_be_done  # <-- FIXED
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
        self.query_expansion_dict = self._build_query_expansion_dict()
        self.persona_weights = self._determine_persona_weights()

        
    def _load_persona(self, persona_file):
        with open(persona_file, "r") as f:
            return json.load(f)
    
    def _build_query_expansion_dict(self):
        """Build a simple query expansion dictionary for common terms."""
        expansion_dict = {
            "ai": ["artificial intelligence", "machine learning", "deep learning", "neural networks"],
            "research": ["study", "analysis", "investigation", "examination"],
            "data": ["information", "dataset", "statistics", "metrics"],
            "technology": ["tech", "innovation", "digital", "computing"],
            "business": ["commercial", "enterprise", "corporate", "industry"],
            "education": ["learning", "teaching", "academic", "training"],
            "health": ["medical", "healthcare", "wellness", "clinical"],
            "finance": ["financial", "economic", "monetary", "fiscal"]
        }
        return expansion_dict
    
    def _determine_persona_weights(self):
        """Determine weighting factors based on persona characteristics."""
        persona_type = self.persona.get("role", "").lower()
        
        # Default weights: [tfidf, keyword_overlap, entity_overlap, heading_level, section_length]
        weights = [0.4, 0.3, 0.2, 0.05, 0.05]
        
        if "researcher" in persona_type or "scientist" in persona_type:
            # Researchers prefer detailed content and specific entities
            weights = [0.3, 0.25, 0.3, 0.1, 0.05]
        elif "student" in persona_type:
            # Students prefer clear headings and structured content
            weights = [0.35, 0.3, 0.15, 0.15, 0.05]
        elif "manager" in persona_type or "executive" in persona_type:
            # Managers prefer summaries and high-level content
            weights = [0.4, 0.3, 0.2, 0.1, 0.0]
        elif "sales" in persona_type or "marketing" in persona_type:
            # Sales/marketing prefer actionable insights and business entities
            weights = [0.35, 0.35, 0.25, 0.05, 0.0]
        
        return weights

    def analyze_documents(self, pdf_paths, outline_extractor_instance):
        """Enhanced document analysis with improved ranking and semantic understanding."""
        start_time = time.time()
        
        all_document_data = []
        document_metadata = []
        
        for pdf_path in pdf_paths:
            outline_data = outline_extractor_instance.extract_outline(pdf_path)
            document_sections = self._create_document_sections_enhanced(outline_data, pdf_path)
            all_document_data.extend(document_sections)
            
            document_metadata.append({
                "file_path": pdf_path,
                "title": outline_data.get("title", ""),
                "total_pages": len(outline_data.get("full_text_by_page", {})),
                "total_headings": len(outline_data.get("headings", []))
            })

        # Enhanced ranking with semantic understanding
        ranked_sections = self._rank_sections_enhanced(all_document_data)
        
        processing_time = time.time() - start_time
        
        results = {
            "metadata": {
                "persona": self.persona,
                "job_to_be_done": self.job_to_be_done,
                "processing_time_seconds": processing_time,
                "total_documents": len(pdf_paths),
                "total_sections": len(all_document_data),
                "document_metadata": document_metadata
            },
            "extracted_sections": ranked_sections[:20]  # Top 20 most relevant sections
        }
        return results

    def _create_document_sections_enhanced(self, outline_data, pdf_path):
        """Create enhanced document sections with better content extraction."""
        sections = []
        full_text_by_page = outline_data["full_text_by_page"]
        headings = outline_data["headings"]
        
        # Group headings by page for better context
        headings_by_page = defaultdict(list)
        for heading in headings:
            headings_by_page[heading["page"]].append(heading)
        
        for i, heading in enumerate(headings):
            page_num = heading["page"]
            
            # Get content for this section (current page and potentially next pages)
            section_content = self._extract_section_content(
                heading, headings, full_text_by_page, i
            )
            
            # Extract additional metadata
            word_count = len(section_content.split())
            
            sections.append({
                "title": heading["text"],
                "level": heading["level"],
                "page": heading["page"],
                "full_text_content": section_content,
                "word_count": word_count,
                "document_source": os.path.basename(pdf_path),
                "section_index": i,
                "importance_rank": 0,
                "font_size": heading.get("font_size", 0),
                "is_bold": heading.get("is_bold", False)
            })
        
        return sections

    def _extract_section_content(self, current_heading, all_headings, full_text_by_page, heading_index):
        """Extract content for a section, considering section boundaries."""
        current_page = current_heading["page"]
        current_level = current_heading["level"]
        
        # Find the next heading of same or higher level to determine section boundary
        section_end_page = None
        for i in range(heading_index + 1, len(all_headings)):
            next_heading = all_headings[i]
            next_level = next_heading["level"]
            
            # If we find a heading of same or higher level, that's our boundary
            if (next_level == "H1") or \
               (current_level == "H1" and next_level in ["H1"]) or \
               (current_level == "H2" and next_level in ["H1", "H2"]) or \
               (current_level == "H3" and next_level in ["H1", "H2", "H3"]):
                section_end_page = next_heading["page"]
                break
        
        # Extract content from current page to section end
        content_parts = []
        
        if section_end_page is None:
            # Section goes to end of document
            for page in range(current_page, max(full_text_by_page.keys()) + 1):
                if page in full_text_by_page:
                    content_parts.append(full_text_by_page[page])
        else:
            # Section has a defined end
            for page in range(current_page, min(section_end_page, max(full_text_by_page.keys()) + 1)):
                if page in full_text_by_page:
                    content_parts.append(full_text_by_page[page])
        
        return " ".join(content_parts)

    def _expand_query(self, query_text):
        """Expand query with synonyms and related terms."""
        expanded_terms = [query_text]
        
        words = query_text.lower().split()
        for word in words:
            if word in self.query_expansion_dict:
                expanded_terms.extend(self.query_expansion_dict[word])
        
        return " ".join(expanded_terms)

    def _preprocess_text(self, text):
        """Enhanced text preprocessing."""
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words and len(word) > 2]
        return " ".join(filtered_tokens)

    def _extract_keywords_enhanced(self, text):
        """Enhanced keyword extraction with better error handling."""
        if not text.strip() or len(text.split()) < 3:
            return []

        keywords_list = []
        
        # Summa keywords with error handling
        try:
            summa_keywords_raw = keywords.keywords(text, words=8, scores=True, split=True)
            if summa_keywords_raw:
                keywords_list.extend([kw[0] for kw in summa_keywords_raw])
        except (IndexError, ValueError, Exception) as e:
            pass
        
        # RAKE keywords
        try:
            r = Rake()
            r.extract_keywords_from_text(text)
            rake_keywords = r.get_ranked_phrases()[:5]  # Top 5 phrases
            if rake_keywords:
                keywords_list.extend(rake_keywords)
        except Exception as e:
            pass
        
        # spaCy-based keywords (noun phrases)
        try:
            doc = nlp(text[:1000])  # Limit text length for performance
            noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
            keywords_list.extend(noun_phrases[:5])
        except Exception as e:
            pass
        
        return list(set(keywords_list))

    def _perform_ner_enhanced(self, text):
        """Enhanced named entity recognition."""
        if not text.strip():
            return []
        
        try:
            # Limit text length for performance
            doc = nlp(text[:2000])
            entities = []
            
            for ent in doc.ents:
                # Filter relevant entity types
                if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"]:
                    entities.append((ent.text.lower(), ent.label_))
            
            return entities
        except Exception as e:
            return []

    def _summarize_text_enhanced(self, text, persona_context=""):
        """Enhanced text summarization with persona awareness."""
        if not text.strip():
            return ""
        
        try:
            # Adjust summary length based on persona
            persona_role = self.persona.get("role", "").lower()
            
            if "executive" in persona_role or "manager" in persona_role:
                word_limit = 30  # Shorter summaries for executives
            elif "researcher" in persona_role:
                word_limit = 80  # Longer summaries for researchers
            else:
                word_limit = 50  # Default
            
            summary = summarizer.summarize(text, words=word_limit)
            
            # If summarization fails, create a simple extractive summary
            if not summary:
                sentences = text.split('.')[:3]  # First 3 sentences
                summary = '. '.join(sentences) + '.'
            
            return summary
        except Exception as e:
            # Fallback: return first few sentences
            sentences = text.split('.')[:2]
            return '. '.join(sentences) + '.' if sentences else text[:200]

    def _rank_sections_enhanced(self, sections):
        """Enhanced section ranking with dynamic weighting and semantic understanding."""
        if not sections:
            return []

        # Expand query for better matching
        expanded_query = self._expand_query(
            self.persona.get("description", "") + " " + self.job_to_be_done.get("task", "")
        )
        query_text = self._preprocess_text(expanded_query)
        
        query_keywords = self._extract_keywords_enhanced(expanded_query)
        query_entities = self._perform_ner_enhanced(expanded_query)
        
        # Prepare section texts
        section_texts = [self._preprocess_text(s["full_text_content"]) for s in sections]
        
        # Handle empty sections
        non_empty_indices = [i for i, text in enumerate(section_texts) if text.strip()]
        if not non_empty_indices:
            for section in sections:
                section["importance_rank"] = 0
                section["refined_text"] = self._summarize_text_enhanced(section["full_text_content"])
            return sections

        # TF-IDF similarity calculation
        non_empty_texts = [section_texts[i] for i in non_empty_indices]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([query_text] + non_empty_texts)
        query_tfidf = tfidf_matrix[0:1]
        section_tfidfs = tfidf_matrix[1:]
        cosine_similarities = cosine_similarity(query_tfidf, section_tfidfs).flatten()
        
        # Calculate scores for each section
        ranked_sections = []
        similarity_idx = 0
        
        for i, section in enumerate(sections):
            section_score = 0.0
            
            if i in non_empty_indices:
                # TF-IDF similarity
                tfidf_score = cosine_similarities[similarity_idx]
                
                # Keyword and entity overlap
                section_keywords = self._extract_keywords_enhanced(section["full_text_content"])
                section_entities = self._perform_ner_enhanced(section["full_text_content"])
                
                keyword_overlap = len(set(query_keywords).intersection(set(section_keywords)))
                entity_overlap = len(set([e[0] for e in query_entities]).intersection(set([e[0] for e in section_entities])))
                
                # Normalize overlaps
                keyword_score = keyword_overlap / max(len(query_keywords), 1)
                entity_score = entity_overlap / max(len(query_entities), 1)
                
                # Heading level importance
                level_map = {"H1": 1.0, "H2": 0.8, "H3": 0.6, "H4": 0.4}
                level_score = level_map.get(section["level"], 0.2)
                
                # Section length (normalized)
                length_score = min(section["word_count"] / 200, 1.0)
                
                # Apply persona-specific weights
                weights = self.persona_weights
                section_score = (
                    tfidf_score * weights[0] +
                    keyword_score * weights[1] +
                    entity_score * weights[2] +
                    level_score * weights[3] +
                    length_score * weights[4]
                )
                
                similarity_idx += 1
            
            section["importance_rank"] = section_score
            section["refined_text"] = self._summarize_text_enhanced(
                section["full_text_content"], 
                self.persona.get("description", "")
            )
            section["relevance_score"] = round(section_score, 4)
            
            ranked_sections.append(section)
        
        # Sort by score and assign final ranks
        ranked_sections.sort(key=lambda x: x["importance_rank"], reverse=True)
        
        for i, section in enumerate(ranked_sections):
            section["final_rank"] = i + 1
        
        return ranked_sections

    def get_analysis_summary(self, results):
        """Generate a summary of the analysis results."""
        total_sections = len(results["extracted_sections"])
        top_sections = results["extracted_sections"][:5]
        
        summary = {
            "total_sections_analyzed": total_sections,
            "top_5_sections": [
                {
                    "title": section["title"],
                    "relevance_score": section["relevance_score"],
                    "document_source": section["document_source"]
                }
                for section in top_sections
            ],
            "processing_time": results["metadata"]["processing_time_seconds"],
            "persona_role": self.persona.get("role", "Unknown"),
            "job_to_be_done": self.job_to_be_done
        }
        
        return summary

if __name__ == "__main__":
    print("Enhanced Document Analyst with improved semantic understanding and persona-driven ranking.")
    print("To test, ensure you have PDF files and call DocumentAnalystEnhanced.analyze_documents(pdf_paths, outline_extractor_instance).")

