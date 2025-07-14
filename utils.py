from langchain_core.tools import tool
import feedparser
import requests
import json
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from typing import List, Dict, Any, Optional


@tool
def search_arxiv(query: str, max_results: int = 10) -> str:
  """Search arXiv for research papers. Returns paper details ncluding title, authors, summary, and URL."""
  try:
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query=all:{query}&start=0&max_results={max_results}&sortBy=lastUpdatedDate&sortOrder=descending"
    
    response = requests.get(base_url + search_query)
    feed = feedparser.parse(response.content)
    
    results = []
    for entry in feed.entries:
        # Extract authors
        authors = [author.name for author in entry.authors] if hasattr(entry, 'authors') else []
        
        # Extract publication date
        pub_date = entry.published if hasattr(entry, 'published') else ""
        
        paper_info = {
            "title": entry.title,
            "authors": authors,
            "summary": entry.summary,
            "url": entry.link,
            "published": pub_date,
            "categories": entry.tags[0].term if hasattr(entry, 'tags') and entry.tags else "",
            "doi": entry.arxiv_doi if hasattr(entry, 'arxiv_doi') else ""
        }
        results.append(paper_info)
    
    return json.dumps(results, indent=2)
  
  except Exception as e:
      return f"Error searching arXiv: {str(e)}"

@tool
def search_pubmed(query: str, max_results: int = 10) -> str:
    """Search PubMed for biomedical research papers. Returns paper details including title, authors, abstract, and PMID."""
    try:
        # PubMed E-utilities API
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        # URL encode the query to handle special characters and spaces
        encoded_query = quote_plus(query)
        
        # First, search for paper IDs
        search_url = f"{base_url}esearch.fcgi?db=pubmed&term={encoded_query}&retmax={max_results}&sort=date&retmode=json"
        
        # Add headers and timeout for better reliability
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; PubMed-Search-Tool/1.0)',
            'Accept': 'application/json'
        }
        
        search_response = requests.get(search_url, headers=headers, timeout=30)
        search_response.raise_for_status()  # Raise exception for HTTP errors
        
        search_data = search_response.json()
        
        # Check for proper response structure
        if 'esearchresult' not in search_data:
            return "Invalid response from PubMed search API"
        
        if 'idlist' not in search_data['esearchresult']:
            return "No results found in PubMed"
        
        ids = search_data['esearchresult']['idlist']
        
        if not ids:
            return "No results found in PubMed"
        
        # Fetch detailed information for each paper
        ids_str = ",".join(ids)
        fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={ids_str}&retmode=xml"
        
        fetch_response = requests.get(fetch_url, headers=headers, timeout=30)
        fetch_response.raise_for_status()
        
        # Parse XML response with better error handling
        try:
            root = ET.fromstring(fetch_response.content)
        except ET.ParseError as e:
            return f"Error parsing XML response: {str(e)}"
        
        results = []
        for article in root.findall('.//PubmedArticle'):
            # Extract title with better handling
            title_elem = article.find('.//ArticleTitle')
            title = "No title"
            if title_elem is not None:
                # Handle case where title might have nested elements
                title = ''.join(title_elem.itertext()).strip()
            
            # Extract abstract with better handling
            abstract_text = ""
            abstract_elems = article.findall('.//AbstractText')
            if abstract_elems:
                abstract_parts = []
                for elem in abstract_elems:
                    # Get label if it exists (like "BACKGROUND:", "METHODS:", etc.)
                    label = elem.get('Label', '')
                    text = ''.join(elem.itertext()).strip()
                    if label and text:
                        abstract_parts.append(f"{label}: {text}")
                    elif text:
                        abstract_parts.append(text)
                abstract_text = " ".join(abstract_parts)
            
            if not abstract_text:
                abstract_text = "No abstract available"
            
            # Extract PMID
            pmid_elem = article.find('.//PMID')
            pmid = pmid_elem.text.strip() if pmid_elem is not None else ""
            
            # Extract authors with better handling
            authors = []
            for author in article.findall('.//Author'):
                lastname_elem = author.find('LastName')
                forename_elem = author.find('ForeName')
                initials_elem = author.find('Initials')
                
                if lastname_elem is not None:
                    lastname = lastname_elem.text.strip()
                    if forename_elem is not None:
                        forename = forename_elem.text.strip()
                        authors.append(f"{forename} {lastname}")
                    elif initials_elem is not None:
                        initials = initials_elem.text.strip()
                        authors.append(f"{initials} {lastname}")
                    else:
                        authors.append(lastname)
            
            # Extract journal information
            journal = ""
            journal_elem = article.find('.//Journal/Title')
            if journal_elem is None:
                journal_elem = article.find('.//Journal/ISOAbbreviation')
            if journal_elem is not None:
                journal = journal_elem.text.strip()
            
            # Extract publication date
            pub_year = ""
            pub_date_elem = article.find('.//PubDate/Year')
            if pub_date_elem is not None:
                pub_year = pub_date_elem.text.strip()
            
            # Extract DOI if available
            doi = ""
            doi_elem = article.find('.//ArticleId[@IdType="doi"]')
            if doi_elem is not None:
                doi = doi_elem.text.strip()
            
            # Truncate abstract if too long
            if len(abstract_text) > 500:
                abstract_text = abstract_text[:500] + "..."
            
            paper_info = {
                "title": title,
                "authors": authors[:5],  # Limit to first 5 authors
                "abstract": abstract_text,
                "pmid": pmid,
                "journal": journal,
                "year": pub_year,
                "doi": doi,
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                "doi_url": f"https://doi.org/{doi}" if doi else ""
            }
            results.append(paper_info)
        
        if not results:
            return "No articles found in the response"
        
        return json.dumps(results, indent=2, ensure_ascii=False)
    
    except requests.exceptions.RequestException as e:
        return f"Network error when accessing PubMed: {str(e)}"
    except json.JSONDecodeError as e:
        return f"Error parsing JSON response: {str(e)}"
    except Exception as e:
        return f"Unexpected error searching PubMed: {str(e)}"

@tool
def search_semantic_scholar(query: str, max_results: int = 10) -> str:
    """
    Search Semantic Scholar for research papers across multiple disciplines.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 10, max: 100)
    
    Returns:
        JSON string with paper details including citation counts
    """
    try:
        # Validate inputs
        if not query or not query.strip():
            return json.dumps({"error": "Query cannot be empty"})
        
        # Limit max_results to reasonable bounds
        max_results = min(max(1, max_results), 100)
        
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        # Headers for better API compliance
        headers = {
            'User-Agent': 'Academic-Research-Tool/1.0'
        }
        
        params = {
            "query": query.strip(),
            "limit": max_results,
            "fields": "title,authors,abstract,year,citationCount,url,venue,externalIds,publicationDate,journal"
        }
        
        # Make request with timeout
        response = requests.get(base_url, params=params, headers=headers, timeout=30)
        
        # Check for HTTP errors
        if response.status_code == 429:
            return json.dumps({"error": "Rate limit exceeded. Please wait before making another request."})
        elif response.status_code != 200:
            return json.dumps({"error": f"HTTP {response.status_code}: {response.text}"})
        
        # Parse JSON response
        try:
            data = response.json()
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON response from API"})
        
        # Check if results exist
        if 'data' not in data or not data['data']:
            return json.dumps({
                "message": "No results found in Semantic Scholar",
                "query": query,
                "results": []
            })
        
        # Process results
        results = []
        for paper in data['data']:
            try:
                # Extract author names safely
                authors = []
                if paper.get('authors'):
                    authors = [author.get('name', 'Unknown') for author in paper['authors']]
                
                # Get external IDs
                external_ids = paper.get('externalIds', {}) or {}
                doi = external_ids.get('DOI', '')
                arxiv_id = external_ids.get('ArXiv', '')
                
                # Clean and format abstract
                abstract = paper.get('abstract', '')
                if abstract and len(abstract) > 500:
                    abstract = abstract[:497] + "..."
                
                # Format venue/journal information
                venue = paper.get('venue', '') or paper.get('journal', {}).get('name', '') if paper.get('journal') else ''
                
                paper_info = {
                    "title": paper.get('title', 'No title available'),
                    "authors": authors,
                    "abstract": abstract or 'No abstract available',
                    "year": paper.get('year') or '',
                    "publication_date": paper.get('publicationDate', ''),
                    "citation_count": paper.get('citationCount', 0),
                    "venue": venue,
                    "url": paper.get('url', ''),
                    "doi": doi,
                    "arxiv_id": arxiv_id
                }
                results.append(paper_info)
                
            except Exception as paper_error:
                # Skip problematic papers but continue processing
                continue
        
        # Return formatted results
        return json.dumps({
            "query": query,
            "total_results": len(results),
            "results": results
        }, indent=2, ensure_ascii=False)
    
    except requests.exceptions.Timeout:
        return json.dumps({"error": "Request timeout. Please try again."})
    except requests.exceptions.ConnectionError:
        return json.dumps({"error": "Connection error. Please check your internet connection."})
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Request error: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


@tool
def search_google_scholar(query: str, max_results: int = 10) -> str:
    """Search Google Scholar for research papers. Note: This requires the scholarly library."""
    try:
        from scholarly import scholarly
        
        results = []
        search_query = scholarly.search_pubs(query)
        
        for i, paper in enumerate(search_query):
            if i >= max_results:
                break
            
            paper_info = {
                "title": paper.get('title', 'No title'),
                "authors": paper.get('author', []),
                "abstract": paper.get('abstract', 'No abstract available'),
                "year": paper.get('year', ''),
                "citation_count": paper.get('num_citations', 0),
                "venue": paper.get('venue', ''),
                "url": paper.get('pub_url', ''),
                "scholar_url": paper.get('scholar_url', '')
            }
            results.append(paper_info)
        
        return json.dumps(results, indent=2)
    
    except ImportError:
        return "Error: scholarly library not installed. Install with: pip install scholarly"
    except Exception as e:
        return f"Error searching Google Scholar: {str(e)}"
      
tools = [search_arxiv, search_pubmed, search_semantic_scholar, search_google_scholar]