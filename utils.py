from langchain_core.tools import tool
import feedparser
import requests
import json
import asyncio

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
        
        # First, search for paper IDs
        search_url = f"{base_url}esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&sort=date&retmode=json"
        search_response = requests.get(search_url)
        search_data = search_response.json()
        
        if 'esearchresult' not in search_data or 'idlist' not in search_data['esearchresult']:
            return "No results found in PubMed"
        
        ids = search_data['esearchresult']['idlist']
        
        if not ids:
            return "No results found in PubMed"
        
        # Fetch detailed information for each paper
        ids_str = ",".join(ids)
        fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={ids_str}&retmode=xml"
        fetch_response = requests.get(fetch_url)
        
        # Parse XML response (simplified parsing)
        import xml.etree.ElementTree as ET
        root = ET.fromstring(fetch_response.content)
        
        results = []
        for article in root.findall('.//PubmedArticle'):
            title_elem = article.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else "No title"
            
            abstract_elem = article.find('.//AbstractText')
            abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
            
            pmid_elem = article.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else ""
            
            # Extract authors
            authors = []
            for author in article.findall('.//Author'):
                lastname = author.find('LastName')
                forename = author.find('ForeName')
                if lastname is not None and forename is not None:
                    authors.append(f"{forename.text} {lastname.text}")
            
            # Extract journal and publication date
            journal_elem = article.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ""
            
            pub_date_elem = article.find('.//PubDate/Year')
            pub_year = pub_date_elem.text if pub_date_elem is not None else ""
            
            paper_info = {
                "title": title,
                "authors": authors,
                "abstract": abstract[:500] + "..." if len(abstract) > 500 else abstract,
                "pmid": pmid,
                "journal": journal,
                "year": pub_year,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            }
            results.append(paper_info)
        
        return json.dumps(results, indent=2)
    
    except Exception as e:
        return f"Error searching PubMed: {str(e)}"

@tool
def search_semantic_scholar(query: str, max_results: int = 10) -> str:
    """Search Semantic Scholar for research papers across multiple disciplines. Returns paper details with citation counts."""
    try:
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,abstract,year,citationCount,url,venue,externalIds"
        }
        
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if 'data' not in data:
            return "No results found in Semantic Scholar"
        
        results = []
        for paper in data['data']:
            # Extract author names
            authors = [author['name'] for author in paper.get('authors', [])]
            
            # Get DOI or other external IDs
            external_ids = paper.get('externalIds', {})
            doi = external_ids.get('DOI', '')
            
            paper_info = {
                "title": paper.get('title', 'No title'),
                "authors": authors,
                "abstract": paper.get('abstract', 'No abstract available'),
                "year": paper.get('year', ''),
                "citation_count": paper.get('citationCount', 0),
                "venue": paper.get('venue', ''),
                "url": paper.get('url', ''),
                "doi": doi
            }
            results.append(paper_info)
        
        return json.dumps(results, indent=2)
    
    except Exception as e:
        return f"Error searching Semantic Scholar: {str(e)}"


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