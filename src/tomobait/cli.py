import argparse

from tomobait.app import run_agent_chat


def main():
    parser = argparse.ArgumentParser(description="TomoBait CLI")
    parser.add_argument("question", type=str, help="The question to ask the agent")
    args = parser.parse_args()
    
    run_agent_chat(args.question)

if __name__ == "__main__":
    main()
