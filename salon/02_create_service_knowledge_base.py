# SALON MULTI-AGENT SYSTEM
# STEP 2: SERVICE KNOWLEDGE BASE FOR RAG
# ***

# Goal: Create a vector database of salon service information
# for the Service Expert agent to answer questions about services

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import yaml

# =============================================================================
# API SETUP
# =============================================================================

# Load API key (adjust path as needed)
try:
    os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']
except:
    print("Note: Set OPENAI_API_KEY environment variable or update credentials path")

# =============================================================================
# SALON SERVICE KNOWLEDGE BASE
# =============================================================================

# This is your salon's service menu and expertise
# Customize this with YOUR salon's actual services, descriptions, and expertise

SALON_SERVICES = """
# HAIRCUT SERVICES

## Men's Haircut - $25
Our men's haircuts include a consultation, precision cut, and style. We specialize in fades, tapers, classic cuts, and modern styles. Every cut includes a hot towel neck shave and styling product application.
- Duration: 30 minutes
- Recommended frequency: Every 3-4 weeks
- Best for: All hair types and styles

## Women's Haircut - $35
A personalized women's haircut including consultation, shampoo, precision cut, and blowout styling. Our stylists are trained in all cutting techniques from blunt cuts to layered styles.
- Duration: 45-60 minutes
- Recommended frequency: Every 6-8 weeks
- Best for: All hair lengths and textures

## Kids Haircut - $18
Patient, kid-friendly haircuts for children 12 and under. We make the experience fun and comfortable for young clients.
- Duration: 20-30 minutes
- Age: 12 and under

## Bang Trim - $10
Quick trim for bangs between regular haircut appointments. Perfect for maintaining your fringe.
- Duration: 10-15 minutes
- Best for: Clients with bangs who need touch-ups

---

# COLOR SERVICES

## Color Retouch - $70
Root touch-up covering new growth. Perfect for maintaining your existing color between full color appointments. Includes processing time and a quick gloss.
- Duration: 60-90 minutes
- Recommended frequency: Every 4-6 weeks
- Best for: Clients with existing color wanting to cover roots

## Full Color - $95
Complete all-over color from roots to ends. Includes consultation, application, processing, and style. We use professional-grade, ammonia-free color options available.
- Duration: 90-120 minutes
- Best for: First-time color, color changes, or gray coverage

## Highlights & Lowlights - $150
Dimensional color using foiling technique. Creates depth and movement in your hair. Price varies based on hair length and density.
- Duration: 2-3 hours
- Recommended frequency: Every 8-12 weeks
- Best for: Adding dimension without full commitment to all-over color

## Balayage - $200
Hand-painted highlighting technique for a natural, sun-kissed look. Creates a softer grow-out than traditional highlights. Includes toner and style.
- Duration: 2.5-3.5 hours
- Recommended frequency: Every 12-16 weeks
- Best for: Low-maintenance color, natural-looking dimension

## Toner - $40
Gloss or toner service to refresh color, neutralize brassiness, or add shine. Can be done as a standalone service or added to any color service.
- Duration: 30-45 minutes
- Best for: Refreshing blonde, neutralizing unwanted tones

## Color Correction - $250+
Complex color services to fix previous color mishaps, remove unwanted color, or achieve dramatic transformations. Consultation required. Price varies significantly based on work needed.
- Duration: 3-6+ hours (may require multiple sessions)
- Consultation required

---

# TREATMENT SERVICES

## Deep Conditioning Treatment - $35
Intensive moisture treatment for dry, damaged, or chemically-treated hair. Includes scalp massage and steam treatment for maximum penetration.
- Duration: 30 minutes
- Recommended frequency: Monthly or as needed
- Best for: Dry, damaged, or color-treated hair

## Keratin Treatment - $250
Smoothing treatment that reduces frizz and cuts styling time in half. Results last 3-5 months. We use formaldehyde-free formulas.
- Duration: 2-3 hours
- Results last: 3-5 months
- Best for: Frizzy, unmanageable hair

## Scalp Treatment - $45
Therapeutic scalp treatment addressing dryness, oiliness, or buildup. Includes exfoliation, treatment mask, and relaxing massage.
- Duration: 30-45 minutes
- Best for: Scalp issues, dandruff, buildup

## Olaplex Treatment - $50
Bond-building treatment that repairs hair from the inside out. Can be added to any color service or done as a standalone treatment.
- Duration: 20-30 minutes (standalone)
- Best for: Damaged, over-processed, or fragile hair

---

# STYLING SERVICES

## Blowout - $45
Professional wash and blow-dry styling. Choose from sleek and straight, bouncy curls, or beachy waves. Includes heat protectant and finishing products.
- Duration: 45 minutes
- Best for: Special occasions or weekly pampering

## Updo - $75
Elegant upstyle for special occasions. Includes consultation, styling, and finishing spray. Add braids, twists, or curls.
- Duration: 60-90 minutes
- Best for: Proms, weddings (guest), formal events

## Bridal Styling - $150
Luxury bridal hair styling including trial run (scheduled separately), day-of styling, and touch-up kit. On-location services available.
- Duration: 90 minutes (trial), 60-90 minutes (day-of)
- Includes: Trial run appointment
- Travel available

## Special Event Styling - $85
Glamorous styling for any special occasion. Includes consultation and style that lasts all night.
- Duration: 60-75 minutes
- Best for: Galas, parties, photo shoots

---

# WAX SERVICES

## Eyebrow Wax - $15
Precise eyebrow shaping using gentle wax. Includes tweezing for detailed shaping and soothing aftercare.
- Duration: 15 minutes
- Recommended frequency: Every 3-4 weeks

## Lip Wax - $10
Quick and gentle upper lip hair removal.
- Duration: 10 minutes
- Recommended frequency: Every 3-4 weeks

## Full Face Wax - $45
Complete facial waxing including eyebrows, lip, chin, and sideburns. Includes soothing treatment afterward.
- Duration: 30 minutes
- Recommended frequency: Every 4-6 weeks

---

# EXTENSION SERVICES

## Tape-In Extensions - $350
Premium tape-in hair extensions for added length and volume. Price includes installation; hair sold separately. We carry a variety of colors and lengths.
- Duration: 90-120 minutes
- Maintenance: Every 6-8 weeks
- Hair lasts: 6-12 months with proper care

## Extension Maintenance - $100
Move-up service for existing tape-in extensions. Includes removal, cleaning, and reinstallation.
- Duration: 60-90 minutes
- Recommended frequency: Every 6-8 weeks

---

# SALON POLICIES

## Cancellation Policy
We require 24-hour notice for cancellations. Late cancellations or no-shows may be subject to a fee equal to 50% of the scheduled service.

## Consultation
We offer free 15-minute consultations for color services, extensions, and any service you'd like to discuss before booking.

## Products We Use
We proudly use and retail professional products including:
- Color: Redken, Wella, Schwarzkopf
- Treatments: Olaplex, K18, Kerastase
- Styling: Moroccan Oil, Bumble and Bumble, R+Co

## Gift Cards
Gift cards are available in any denomination and never expire. Perfect for any occasion!

## Loyalty Program
Earn 1 point for every dollar spent. 100 points = $10 off your next service. VIP members (500+ lifetime points) receive early access to promotions and exclusive perks.
"""

# =============================================================================
# CREATE VECTOR DATABASE
# =============================================================================

def create_service_knowledge_base(
    content: str = SALON_SERVICES,
    persist_directory: str = "data/salon_services_vectordb",
    chunk_size: int = 500,
    chunk_overlap: int = 50
):
    """Create a Chroma vector database from salon service content"""

    # Create documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n---\n", "\n\n", "\n", " "]
    )

    docs = text_splitter.create_documents([content])

    print(f"Created {len(docs)} document chunks")

    # Create embeddings
    embedding_function = OpenAIEmbeddings(
        model='text-embedding-ada-002'
    )

    # Create and persist vector store
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_function,
        persist_directory=persist_directory
    )

    print(f"Vector database created at: {persist_directory}")

    return vectorstore


# =============================================================================
# TEST RETRIEVAL
# =============================================================================

def test_retrieval(persist_directory: str = "data/salon_services_vectordb"):
    """Test the vector database retrieval"""

    embedding_function = OpenAIEmbeddings(model='text-embedding-ada-002')

    vectorstore = Chroma(
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )

    test_queries = [
        "What color services do you offer?",
        "How much does a balayage cost?",
        "What's the difference between highlights and balayage?",
        "Do you have treatments for damaged hair?",
        "What's your cancellation policy?",
    ]

    print("\n" + "="*60)
    print("TESTING RETRIEVAL")
    print("="*60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vectorstore.similarity_search(query, k=2)
        for i, doc in enumerate(results):
            print(f"  Result {i+1}: {doc.page_content[:150]}...")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    os.makedirs('data', exist_ok=True)

    print("Creating salon service knowledge base...")
    vectorstore = create_service_knowledge_base()

    print("\nTesting retrieval...")
    test_retrieval()

    print("\n" + "="*60)
    print("SERVICE KNOWLEDGE BASE READY!")
    print("="*60)
