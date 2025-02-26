#!/usr/bin/env python3
import argparse
import requests
import json
import os
import sys
import subprocess
import textwrap
from tabulate import tabulate
from colorama import Fore, Style, init

# Initialize colorama
init()

def nl_to_sql(query):
    """Convert natural language query to SQL using the Trino-AI service"""
    print(f"{Fore.CYAN}Converting natural language query to SQL...{Style.RESET_ALL}")
    
    try:
        response = requests.post(
            "http://localhost:5001/utility/nl2sql",
            headers={"Content-Type": "application/json"},
            json={"query": query}
        )
        
        if response.status_code != 200:
            print(f"{Fore.RED}Error: {response.status_code} - {response.text}{Style.RESET_ALL}")
            return None, None, None
        
        result = response.json()
        sql_query = result.get("sql_query", "")
        explanation = result.get("explanation", "")
        context = result.get("context_used", "")
        
        return sql_query, explanation, context
    
    except Exception as e:
        print(f"{Fore.RED}Error connecting to Trino-AI service: {str(e)}{Style.RESET_ALL}")
        return None, None, None

def execute_sql(sql_query):
    """Execute SQL query using Trino CLI"""
    print(f"{Fore.CYAN}Executing SQL query...{Style.RESET_ALL}")
    
    try:
        # Format the SQL query for better readability
        formatted_sql = sql_query.replace('\n', ' ').strip()
        
        # Execute the query using Trino CLI
        cmd = [
            "docker", "exec", "-i", "trino", 
            "trino", "--server", "localhost:8080", 
            "--catalog", "iceberg", 
            "--schema", "iceberg", 
            "--output-format", "CSV", 
            "--execute", formatted_sql
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"{Fore.RED}Error executing SQL query: {result.stderr}{Style.RESET_ALL}")
            return None
        
        # Parse CSV output
        lines = result.stdout.strip().split('\n')
        if not lines:
            return []
        
        # Parse header and data
        header = lines[0].split(',')
        header = [h.strip('"') for h in header]
        
        data = []
        for line in lines[1:]:
            if line.strip():
                values = line.split(',')
                values = [v.strip('"') for v in values]
                data.append(values)
        
        return {"header": header, "data": data}
    
    except Exception as e:
        print(f"{Fore.RED}Error executing SQL query: {str(e)}{Style.RESET_ALL}")
        return None

def format_results(results):
    """Format results for display"""
    if not results or "header" not in results or "data" not in results:
        return "No results found."
    
    return tabulate(results["data"], headers=results["header"], tablefmt="pretty")

def main():
    parser = argparse.ArgumentParser(description="Run natural language queries against Trino using Trino-AI")
    parser.add_argument("query", nargs="?", help="Natural language query to execute")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()
    
    if not args.query and not args.interactive:
        parser.print_help()
        sys.exit(1)
    
    def process_query(query):
        print(f"\n{Fore.GREEN}Query: {query}{Style.RESET_ALL}")
        
        # Convert natural language to SQL
        sql_query, explanation, context = nl_to_sql(query)
        if not sql_query:
            return
        
        # Display the generated SQL
        print(f"\n{Fore.YELLOW}Generated SQL:{Style.RESET_ALL}")
        print(textwrap.indent(sql_query, "  "))
        
        # Execute the SQL query
        results = execute_sql(sql_query)
        if not results:
            return
        
        # Display the results
        print(f"\n{Fore.YELLOW}Results:{Style.RESET_ALL}")
        print(format_results(results))
        
        # Display the explanation
        if explanation:
            print(f"\n{Fore.YELLOW}Explanation:{Style.RESET_ALL}")
            print(textwrap.indent(explanation, "  "))
    
    if args.interactive:
        print(f"{Fore.GREEN}=== Trino-AI Natural Language Query Interface ==={Style.RESET_ALL}")
        print(f"{Fore.GREEN}Type your questions in natural language. Type 'exit' or 'quit' to end.{Style.RESET_ALL}")
        
        while True:
            try:
                query = input(f"\n{Fore.CYAN}> {Style.RESET_ALL}")
                if query.lower() in ["exit", "quit", "q"]:
                    break
                
                if query.strip():
                    process_query(query)
            
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            
            except Exception as e:
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    else:
        process_query(args.query)

if __name__ == "__main__":
    main() 