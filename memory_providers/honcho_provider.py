"""
Honcho Memory Provider

Dialectical user modeling (psychology-based).

From plastic-labs/honcho:
    "Honcho uses an entity-centric model where both users and agents are represented as 'peers'."
    
Architecture:
    1. Peer Model: Users and agents are both "peers" with their own state
    2. Session Management: Multi-participant sessions with observation settings
    3. Dialectical Reasoning: Builds psychological model of user
    4. Representation: User "cards" that capture personality/preferences
    
Key differentiator:
    - NOT a fact store or knowledge graph
    - Builds a PSYCHOLOGICAL MODEL of the user
    - Answers "what kind of person is this?" not "what facts do I know?"
    - Useful for personalization, not temporal reasoning

From their research (blog.plasticlabs.ai):
    - Defined "Pareto Frontier" of agent memory
    - Focus on retention, trust, data moats
    - Dialectical approach inspired by cognitive science

This is ORTHOGONAL to PIE:
    - PIE: "What happened to Project X?" (temporal facts)
    - Honcho: "How should I communicate with this user?" (user psychology)
"""

from __future__ import annotations
import logging
import os
from typing import Any

from .interface import MemoryProvider, MemoryProviderConfig, SearchResult, MemoryStats

logger = logging.getLogger("honcho.provider")


class HonchoProvider(MemoryProvider):
    """
    Honcho memory provider.
    
    Focuses on user modeling rather than fact storage.
    Best for: personalization, communication style adaptation
    Less suited for: temporal reasoning, fact recall
    """
    
    name = "honcho"
    
    def __init__(self, config: MemoryProviderConfig | None = None):
        super().__init__(config)
        
        self.api_key = config.api_key if config else os.environ.get("HONCHO_API_KEY")
        self._use_cloud = bool(self.api_key)
        self._client = None
        
        # Local state
        self._messages: list[dict] = []
        self._user_representation: dict = {}
        self._preferences: list[str] = []
        
        if self._use_cloud:
            self._init_cloud_client()
    
    def _init_cloud_client(self):
        """Initialize Honcho Cloud client."""
        try:
            from honcho import Honcho
            self._client = Honcho(workspace_id="pie_benchmark")
            logger.info("Initialized Honcho Cloud client")
        except ImportError:
            logger.info("honcho-ai not installed, using local simulation")
            self._use_cloud = False
    
    def ingest(self, sessions: list[list[dict]], dates: list[str] | None = None) -> None:
        """
        Ingest sessions and build user model.
        
        Honcho's approach:
        1. Store messages in sessions
        2. Run "deriver" to build representations
        3. Generate user "cards" with psychological profile
        """
        if self._use_cloud and self._client:
            self._ingest_cloud(sessions, dates)
        else:
            self._ingest_local(sessions, dates)
    
    def _ingest_cloud(self, sessions: list[list[dict]], dates: list[str] | None):
        """Ingest via Honcho API."""
        try:
            # Create peers
            user = self._client.peer("user")
            assistant = self._client.peer("assistant")
            
            for i, session_turns in enumerate(sessions):
                session = self._client.session(f"session_{i}")
                
                messages = []
                for turn in session_turns:
                    peer = user if turn.get("role") == "user" else assistant
                    messages.append(peer.message(turn.get("content", "")))
                
                if messages:
                    session.add_messages(messages)
            
            logger.info(f"Ingested {len(sessions)} sessions to Honcho Cloud")
            
        except Exception as e:
            logger.warning(f"Honcho Cloud failed: {e}")
            self._use_cloud = False
            self._ingest_local(sessions, dates)
    
    def _ingest_local(self, sessions: list[list[dict]], dates: list[str] | None):
        """
        Local simulation of Honcho's dialectical modeling.
        
        Key insight: Honcho builds a psychological profile, not a fact store.
        We simulate this by extracting:
        1. Communication preferences
        2. Personality traits
        3. Values and beliefs
        4. Learning styles
        """
        from openai import OpenAI
        client = OpenAI()
        
        # Store all messages
        for i, session in enumerate(sessions):
            date = dates[i] if dates and i < len(dates) else f"2025-01-{i+1:02d}"
            for turn in session:
                self._messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                    "date": date,
                    "session_idx": i,
                })
        
        # Build user representation (Honcho's "dialectical" approach)
        # Sample messages to build profile
        user_messages = [m["content"] for m in self._messages if m["role"] == "user"][:50]
        
        if user_messages:
            sample_text = "\n".join(user_messages[:20])
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": f"""Analyze this user's communication style and preferences.

User messages:
{sample_text[:5000]}

Create a psychological profile with:
1. Communication style (formal/casual, verbose/concise)
2. Apparent expertise areas
3. Values and priorities
4. Preferred interaction patterns
5. Learning style indicators

Return as JSON: {{"profile": {{"communication_style": "...", "expertise": [...], "values": [...], "interaction_preferences": [...], "learning_style": "..."}}}}"""
                    }],
                    response_format={"type": "json_object"},
                    max_tokens=500,
                )
                
                import json
                result = json.loads(response.choices[0].message.content)
                self._user_representation = result.get("profile", {})
                
            except Exception as e:
                logger.warning(f"Profile generation failed: {e}")
        
        # Extract preferences (things the user explicitly prefers)
        if user_messages:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": f"""Extract explicit user preferences from these messages.
Look for statements like "I prefer...", "I like...", "I don't want...", etc.

Messages:
{sample_text[:4000]}

Return as JSON: {{"preferences": ["preference 1", "preference 2", ...]}}"""
                    }],
                    response_format={"type": "json_object"},
                    max_tokens=300,
                )
                
                import json
                result = json.loads(response.choices[0].message.content)
                self._preferences = result.get("preferences", [])
                
            except Exception as e:
                logger.warning(f"Preference extraction failed: {e}")
        
        logger.info(f"Ingested {len(self._messages)} messages, built user profile")
    
    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Search using Honcho's approach.
        
        Honcho is less about retrieval and more about understanding.
        We return both relevant messages AND user profile context.
        """
        if self._use_cloud and self._client:
            return self._search_cloud(query, top_k)
        else:
            return self._search_local(query, top_k)
    
    def _search_cloud(self, query: str, top_k: int) -> list[SearchResult]:
        """Search via Honcho API."""
        try:
            user = self._client.peer("user")
            results = user.search(query)
            
            return [
                SearchResult(
                    content=str(r),
                    score=0.5,
                    metadata={},
                )
                for r in results[:top_k]
            ]
        except Exception as e:
            logger.error(f"Honcho search failed: {e}")
            return self._search_local(query, top_k)
    
    def _search_local(self, query: str, top_k: int) -> list[SearchResult]:
        """Local search with user profile context."""
        results = []
        
        # Always include user profile
        if self._user_representation:
            profile_str = "\n".join([
                f"{k}: {v}" for k, v in self._user_representation.items()
            ])
            results.append(SearchResult(
                content=f"User Profile:\n{profile_str}",
                score=1.0,
                metadata={"type": "profile"},
            ))
        
        # Include relevant preferences
        if self._preferences:
            prefs_str = "\n".join([f"- {p}" for p in self._preferences])
            results.append(SearchResult(
                content=f"User Preferences:\n{prefs_str}",
                score=0.9,
                metadata={"type": "preferences"},
            ))
        
        # Simple keyword search through messages
        query_lower = query.lower()
        keywords = query_lower.split()
        
        scored_messages = []
        for msg in self._messages:
            if msg["role"] != "user":
                continue
            content = msg["content"].lower()
            score = sum(1 for kw in keywords if kw in content) / len(keywords) if keywords else 0
            if score > 0:
                scored_messages.append((msg, score))
        
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        
        for msg, score in scored_messages[:top_k - len(results)]:
            results.append(SearchResult(
                content=msg["content"],
                score=score,
                metadata={"date": msg.get("date"), "type": "message"},
            ))
        
        return results
    
    def answer(self, question: str, question_date: str | None = None) -> str:
        """
        Answer using Honcho's user-aware approach.
        
        Key difference: Honcho prioritizes HOW to answer based on user profile,
        not just WHAT to answer based on facts.
        """
        results = self.search(question, top_k=10)
        
        # Separate profile from content
        profile_context = ""
        content_context = ""
        
        for r in results:
            if r.metadata.get("type") in ("profile", "preferences"):
                profile_context += r.content + "\n"
            else:
                content_context += f"[{r.metadata.get('date', '?')}] {r.content}\n"
        
        from openai import OpenAI
        client = OpenAI()
        
        # Honcho-style prompt: adapt response to user profile
        system_prompt = f"""You are a helpful AI assistant that adapts to the user's style.

{profile_context}

Use this understanding of the user to tailor your response appropriately."""
        
        user_prompt = f"""Context from past conversations:
{content_context}

Question: {question}

Answer in a way that matches the user's communication preferences:"""
        
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
        )
        
        return response.choices[0].message.content.strip()
    
    def stats(self) -> MemoryStats:
        return MemoryStats(
            num_memories=len(self._messages),
            extra={
                "has_profile": bool(self._user_representation),
                "num_preferences": len(self._preferences),
            },
        )
    
    def clear(self) -> None:
        self._messages = []
        self._user_representation = {}
        self._preferences = []
