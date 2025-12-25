# SALON MULTI-AGENT SYSTEM
# INTERACTIVE DEMO
# ***

"""
Interactive demo of the Salon Multi-Agent System.
Run this after completing setup (00_run_setup.py)
"""

from langchain_core.messages import HumanMessage, AIMessage
import os
import sys

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Import the salon copilot
from salon_multi_agent_system import salon_copilot, get_last_ai_message

# =============================================================================
# EXAMPLE QUERIES
# =============================================================================

EXAMPLE_QUERIES = """
EXAMPLE QUERIES YOU CAN TRY:

SERVICE QUESTIONS:
- "What hair color services do you offer?"
- "How much does a balayage cost and how long does it take?"
- "What treatments do you recommend for damaged hair?"
- "What's your cancellation policy?"

BUSINESS INTELLIGENCE:
- "What are our top 5 services by revenue?"
- "Show me monthly revenue trends by category"
- "Which day of the week has the most appointments?"
- "What's the average spend per customer visit?"
- "How many new customers did we get last month?"

CUSTOMER SCORING:
- "Find customers at high risk of churning"
- "Who are our VIP customers?"
- "Find customers who only get haircuts - they might be interested in color"
- "Which customers have high upsell potential?"

MARKETING EMAILS:
- "Write a re-engagement email for customers who haven't visited in 60+ days"
- "Create a promotional email for our new keratin treatment"
- "Write a thank-you email for our VIP customers"
- "Create a summer promotion email for color services"

MULTI-AGENT WORKFLOWS:
- "Find high-value customers who haven't visited recently and write them a VIP appreciation email"
- "Identify customers who only get haircuts, then write an email introducing our color services"
- "What are our most popular services? Write an email highlighting our signature offerings"
"""

def print_response(result):
    """Print agent responses nicely"""
    print("\n" + "-"*60)

    for msg in result.get('messages', []):
        if isinstance(msg, AIMessage) and msg.name:
            print(f"\n[{msg.name}]:")
            print("-" * 40)
            print(msg.content)
            print()

def main():
    print("\n" + "="*60)
    print("SALON MULTI-AGENT SYSTEM - INTERACTIVE DEMO")
    print("="*60)
    print(EXAMPLE_QUERIES)
    print("="*60)
    print("\nType 'quit' or 'exit' to end the session")
    print("Type 'examples' to see example queries again")
    print("="*60)

    while True:
        try:
            user_input = input("\n\nYour question: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if user_input.lower() == 'examples':
                print(EXAMPLE_QUERIES)
                continue

            print("\nProcessing... (this may take a moment)")

            result = salon_copilot.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"recursion_limit": 15}
            )

            print_response(result)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or check your setup.")

if __name__ == "__main__":
    main()
