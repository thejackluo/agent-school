"""
Doc Ingester - Scrapes help centers and converts to LLM-friendly documentation

Uses Browser Use to navigate help pages and LLM to convert raw content
into structured, agent-friendly documentation.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from .doc_store import DocStore, DocEntry


class DocIngester:
    """
    Ingests help center documentation and converts to LLM-friendly format.
    
    Process:
    1. Navigate to help center URL
    2. Extract page content
    3. Use LLM to structure and summarize
    4. Store in DocStore for retrieval
    """
    
    def __init__(
        self,
        llm_provider: str = "anthropic",
        doc_store: Optional[DocStore] = None,
    ):
        self.llm_provider = llm_provider
        self.doc_store = doc_store or DocStore()
        self._client = None
    
    def _get_llm_client(self):
        """Get LLM client for doc processing"""
        if self._client is None:
            if self.llm_provider == "anthropic":
                from anthropic import Anthropic
                self._client = Anthropic()
            else:
                from openai import OpenAI
                self._client = OpenAI()
        return self._client
    
    async def ingest_help_center(
        self,
        url: str,
        domain: str,
        max_pages: int = 10,
    ) -> List[DocEntry]:
        """
        Ingest help center pages starting from URL.
        
        Args:
            url: Starting URL for help center
            domain: Domain being documented (e.g., "mail.google.com")
            max_pages: Maximum pages to ingest
            
        Returns:
            List of ingested DocEntry objects
        """
        try:
            from browser_use import Agent, Browser
        except ImportError:
            raise ImportError("browser-use is required. Install with: uv add browser-use")
        
        entries = []
        
        # Use Browser Use to extract page content
        browser = Browser()
        agent = Agent(
            task=f"""Navigate to {url} and extract the main help content.
            Return the page title and main content text.
            Format as JSON: {{"title": "...", "content": "..."}}""",
            browser=browser,
        )
        
        try:
            result = await agent.run()
            
            # Parse result and create entry
            content = self._extract_content_from_result(result)
            
            if content:
                # Process with LLM
                llm_docs = self._convert_to_llm_docs(content, domain)
                
                entry = DocEntry(
                    domain=domain,
                    title=llm_docs.get("title", "Help Documentation"),
                    content=llm_docs.get("content", content),
                    source_url=url,
                    ingested_at=datetime.now(),
                    tags=llm_docs.get("tags", []),
                )
                
                self.doc_store.store(entry)
                entries.append(entry)
        finally:
            await browser.close()
        
        return entries
    
    def ingest_text(
        self,
        text: str,
        domain: str,
        title: str,
        source_url: str = "manual",
        tags: Optional[List[str]] = None,
    ) -> DocEntry:
        """
        Ingest raw text documentation directly.
        
        Useful for manually providing documentation without scraping.
        
        Args:
            text: Raw documentation text
            domain: Domain being documented
            title: Title for the documentation
            source_url: Where this came from (or "manual")
            tags: Optional tags for categorization
            
        Returns:
            Stored DocEntry
        """
        # Process with LLM to make it agent-friendly
        llm_docs = self._convert_to_llm_docs(text, domain)
        
        entry = DocEntry(
            domain=domain,
            title=title,
            content=llm_docs.get("content", text),
            source_url=source_url,
            ingested_at=datetime.now(),
            tags=tags or llm_docs.get("tags", []),
        )
        
        self.doc_store.store(entry)
        return entry
    
    def _extract_content_from_result(self, result: Any) -> Optional[str]:
        """Extract text content from Browser Use result"""
        if result is None:
            return None
        
        # Handle different result formats
        if hasattr(result, 'final_result'):
            return str(result.final_result)
        elif hasattr(result, 'extracted_content'):
            return str(result.extracted_content)
        elif isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return result.get("content") or result.get("text") or str(result)
        
        return str(result)
    
    def _convert_to_llm_docs(self, raw_content: str, domain: str) -> Dict[str, Any]:
        """
        Use LLM to convert raw content to structured, agent-friendly docs.
        
        The output format is designed to be useful during exploration:
        - Clear step-by-step instructions
        - UI element names/labels
        - Common actions and their results
        """
        client = self._get_llm_client()
        
        prompt = f"""You are a documentation processor. Convert this help content into a clear, 
structured guide for an AI agent that will automate browser tasks on {domain}.

The output should:
1. Focus on step-by-step procedures
2. Include UI element names (button text, link text, menu items)
3. Describe what each action does
4. Note any prerequisites or requirements
5. Be concise but complete

Raw content:
{raw_content[:4000]}  # Truncate to avoid token limits

Output as JSON with:
- "title": Clear title for this procedure
- "content": Markdown-formatted guide optimized for AI automation
- "tags": List of relevant tags (e.g., ["email", "compose", "attachment"])
"""
        
        try:
            if self.llm_provider == "anthropic":
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}]
                )
                result_text = response.content[0].text
            else:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}]
                )
                result_text = response.choices[0].message.content
            
            # Parse JSON from response
            import json
            # Try to find JSON in response
            if "{" in result_text:
                json_start = result_text.index("{")
                json_end = result_text.rindex("}") + 1
                return json.loads(result_text[json_start:json_end])
            
            return {"content": raw_content, "title": "Documentation", "tags": []}
            
        except Exception as e:
            print(f"Warning: LLM processing failed: {e}")
            return {"content": raw_content, "title": "Documentation", "tags": []}
    
    def ingest_google_docs_help(self) -> List[DocEntry]:
        """Pre-defined ingestion for Google Workspace documentation"""
        # We can bootstrap with known documentation
        entries = []
        
        google_docs = [
            {
                "domain": "mail.google.com",
                "title": "Composing an Email in Gmail",
                "content": """# How to Compose an Email in Gmail

## Steps:
1. Click the **Compose** button (top-left corner, prominent blue/red button)
2. In the compose window that appears:
   - **To** field: Enter recipient email address
   - **Subject** field: Enter email subject
   - **Body**: Type your message in the large text area
3. Click **Send** button (blue button at bottom of compose window)

## UI Elements:
- "Compose" button - Opens new email window
- "To" input field - Recipient address
- "Subject" input field - Email subject line
- "Send" button - Sends the email
- "Discard" (trash icon) - Deletes draft

## Common Variations:
- **CC**: Click "Cc" link to add carbon copy recipients
- **BCC**: Click "Bcc" link for blind carbon copy
- **Attachments**: Click paperclip icon to attach files
- **Formatting**: Use toolbar for bold, italic, links, etc.
""",
                "source_url": "https://support.google.com/mail/answer/8220",
                "tags": ["email", "compose", "send", "gmail"],
            },
            {
                "domain": "docs.google.com",
                "title": "Creating a New Google Doc",
                "content": """# How to Create a New Google Doc

## Steps:
1. Navigate to docs.google.com
2. Click the **+ Blank** button (or choose a template)
3. A new document opens with "Untitled document" as the name
4. Click on "Untitled document" text to rename it
5. Start typing in the document body

## UI Elements:
- "+ Blank" - Creates new blank document
- "Untitled document" - Click to rename
- Document body - Main editing area
- Menu bar - File, Edit, View, Insert, Format, Tools, Extensions, Help

## Common Actions:
- **Save**: Automatic (shows "Saved" in top bar)
- **Share**: Click blue "Share" button (top-right)
- **Download**: File > Download > Choose format
""",
                "source_url": "https://support.google.com/docs/answer/7068618",
                "tags": ["docs", "create", "document"],
            },
            {
                "domain": "sheets.google.com",
                "title": "Creating and Editing Google Sheets",
                "content": """# How to Work with Google Sheets

## Creating a New Sheet:
1. Navigate to sheets.google.com
2. Click the **+ Blank** button
3. Click "Untitled spreadsheet" to rename

## Entering Data:
1. Click on any cell to select it
2. Type your data
3. Press Enter to move to next row, Tab for next column

## UI Elements:
- Cell grid - Main data area (columns A, B, C... rows 1, 2, 3...)
- Formula bar - Shows/edits cell contents
- Sheet tabs - Bottom of screen, switch between sheets
- "+" button - Add new sheet tab

## Common Formulas:
- **SUM**: =SUM(A1:A10) - Adds numbers
- **AVERAGE**: =AVERAGE(B1:B10) - Calculates mean
- **COUNT**: =COUNT(C1:C10) - Counts numbers

## Formatting:
- Select cells, then use toolbar for:
  - Bold (Ctrl+B)
  - Colors (paint bucket icon)
  - Borders (grid icon)
""",
                "source_url": "https://support.google.com/docs/answer/6000292",
                "tags": ["sheets", "spreadsheet", "data", "formula"],
            },
        ]
        
        for doc in google_docs:
            entry = DocEntry(
                domain=doc["domain"],
                title=doc["title"],
                content=doc["content"],
                source_url=doc["source_url"],
                ingested_at=datetime.now(),
                tags=doc["tags"],
            )
            self.doc_store.store(entry)
            entries.append(entry)
        
        return entries
