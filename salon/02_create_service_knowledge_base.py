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

## Men's Haircut - $25-$30
Our men's haircuts include a consultation, precision cut, and style. We specialize in fades, tapers, classic cuts, and modern styles.
- Duration: 30 minutes
- Recommended frequency: Every 3-4 weeks
- Best for: All hair types and styles

## Women's Haircut - $35-$50
A personalized women's haircut including consultation, shampoo, precision cut, and styling. Our stylists are trained in all cutting techniques from blunt cuts to layered styles.
- Duration: 45-60 minutes
- Recommended frequency: Every 6-8 weeks
- Best for: All hair lengths and textures

## Child Haircut - $18-$20
Patient, kid-friendly haircuts for children. We make the experience fun and comfortable for young clients.
- Duration: 20-30 minutes
- Best for: Children

## Shampoo, Haircut & Blow Dry - $60
Complete service including shampoo, precision haircut, and professional blow dry styling.
- Duration: 60-75 minutes
- Best for: Full service appointment

---

# COLOR SERVICES

## Color Retouch - $70-$90
Root touch-up covering new growth. Perfect for maintaining your existing color between full color appointments. Includes processing time.
- Duration: 60-90 minutes
- Recommended frequency: Every 4-6 weeks
- Best for: Clients with existing color wanting to cover roots

## Toner - $40
Gloss or toner service to refresh color, neutralize brassiness, or add shine. Can be done as a standalone service or added to any color service.
- Duration: 30-45 minutes
- Best for: Refreshing blonde, neutralizing unwanted tones

---

# FOILS / HIGHLIGHTS

## Highlights - $150
Classic foil highlights to add dimension and brightness to your hair. Price varies based on hair length and density.
- Duration: 2-3 hours
- Recommended frequency: Every 8-12 weeks
- Best for: Adding brightness and dimension

## Highlights & Low Lights - $150+
Combination of highlights and lowlights using foiling technique. Creates beautiful depth and movement in your hair.
- Duration: 2-3 hours
- Recommended frequency: Every 8-12 weeks
- Best for: Multi-dimensional color, adding depth and brightness

## Highlights and Root Color - $180+
Full highlights combined with root color coverage. Perfect for clients who want both dimension and gray coverage.
- Duration: 2.5-3.5 hours
- Best for: Clients needing both highlights and root coverage

---

# TREATMENT SERVICES

## Deep Conditioning Treatment - $35
Intensive moisture treatment for dry, damaged, or chemically-treated hair. Includes scalp massage and steam treatment for maximum penetration.
- Duration: 30 minutes
- Recommended frequency: Monthly or as needed
- Best for: Dry, damaged, or color-treated hair

---

# SMOOTHING TREATMENTS

## Brazilian Blow Out - $150-$250
Professional smoothing treatment that eliminates frizz, smooths the cuticle, and creates shiny, manageable hair. Results improve with each wash and last up to 12 weeks.
- Duration: 2-3 hours
- Results last: 10-12 weeks
- Best for: Frizzy, curly, or unmanageable hair
- Note: Reduces styling time significantly

---

# STYLING SERVICES

## Shampoo & Blow Dry - $15-$70
Professional wash and blow-dry styling. Choose from sleek and straight, bouncy volume, or soft waves. Includes heat protectant and finishing products.
- Duration: 30-45 minutes
- Best for: Special occasions, weekly pampering, or between haircuts

## Blow Out - $100
Premium professional blow-dry styling service for special occasions. Includes deep conditioning, scalp massage, and expert styling.
- Duration: 45-60 minutes
- Best for: Special events, photo shoots, date nights

## Updo - $75
Elegant upstyle for special occasions. Includes consultation, styling, and finishing spray. Add braids, twists, or curls.
- Duration: 60-90 minutes
- Best for: Proms, weddings (guest), formal events

---

# ROLLER SETS

## Shampoo & Set - $25
Classic roller set service including shampoo, setting on rollers, time under the dryer, and style out. Creates beautiful, long-lasting curls and volume.
- Duration: 60-90 minutes
- Best for: Clients who love classic curls, volume, and long-lasting styles
- Great for: Special occasions or weekly styling

---

# WAX SERVICES

## Eyebrow Wax - $15
Precise eyebrow shaping using gentle wax. Includes tweezing for detailed shaping and soothing aftercare.
- Duration: 15 minutes
- Recommended frequency: Every 3-4 weeks

## Lip Wax - $10-$15
Quick and gentle upper lip hair removal.
- Duration: 10 minutes
- Recommended frequency: Every 3-4 weeks

## Chin Wax - $15
Gentle chin hair removal with soothing aftercare.
- Duration: 10-15 minutes
- Recommended frequency: Every 3-4 weeks

## Full Face Wax - $45
Complete facial waxing including eyebrows, lip, chin, and sideburns. Includes soothing treatment afterward.
- Duration: 30 minutes
- Recommended frequency: Every 4-6 weeks

## Eyebrow Tint - $15-$20
Semi-permanent eyebrow coloring to enhance and define your brows. Lasts 4-6 weeks.
- Duration: 15-20 minutes
- Best for: Lighter brows, adding definition

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
