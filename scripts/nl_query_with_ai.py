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
        
        # Fix the SQL query if needed
        sql_query = fix_sql_query(sql_query)
        
        return sql_query, explanation, context
    
    except Exception as e:
        print(f"{Fore.RED}Error connecting to Trino-AI service: {str(e)}{Style.RESET_ALL}")
        return None, None, None

def fix_sql_query(sql_query):
    """Fix common issues with generated SQL queries"""
    # Replace references to non-existent orders table
    if "iceberg.iceberg.orders" in sql_query:
        print(f"{Fore.YELLOW}Fixing SQL query: Removing reference to non-existent orders table{Style.RESET_ALL}")
        
        # If the query joins customers, orders, and sales, simplify to join customers and sales directly
        if "iceberg.iceberg.customers" in sql_query and "iceberg.iceberg.sales" in sql_query:
            # Remove the orders join and fix the join condition
            sql_query = sql_query.replace(
                "JOIN iceberg.iceberg.orders ON iceberg.iceberg.customers.customer_id = iceberg.iceberg.orders.customer_id JOIN iceberg.iceberg.sales ON iceberg.iceberg.orders.order_id = iceberg.iceberg.sales.order_id",
                "JOIN iceberg.iceberg.sales ON iceberg.iceberg.customers.customer_id = iceberg.iceberg.sales.customer_id"
            )
    
    # Fix incorrect FROM clause
    if "FROM iceberg.iceberg" in sql_query:
        print(f"{Fore.YELLOW}Fixing SQL query: Correcting FROM clause{Style.RESET_ALL}")
        sql_query = sql_query.replace(
            "FROM iceberg.iceberg",
            "FROM iceberg.iceberg.customers"
        )
        
        # Also fix any incorrect JOIN conditions
        sql_query = sql_query.replace(
            "iceberg.iceberg.customers.customer_id = iceberg.iceberg.customer_id",
            "iceberg.iceberg.customers.customer_id = iceberg.iceberg.sales.customer_id"
        )
    
    # If we're missing the sales table in the FROM clause but using it in SELECT
    if "iceberg.iceberg.sales.gross_amount" in sql_query and "JOIN iceberg.iceberg.sales" not in sql_query:
        print(f"{Fore.YELLOW}Fixing SQL query: Adding missing JOIN to sales table{Style.RESET_ALL}")
        sql_query = sql_query.replace(
            "FROM iceberg.iceberg.customers",
            "FROM iceberg.iceberg.customers JOIN iceberg.iceberg.sales ON iceberg.iceberg.customers.customer_id = iceberg.iceberg.sales.customer_id"
        )
    
    return sql_query

def execute_sql(sql_query, use_ai_functions=False):
    """Execute SQL query using Trino CLI"""
    print(f"{Fore.CYAN}Executing SQL query...{Style.RESET_ALL}")
    
    try:
        # Format the SQL query for better readability
        formatted_sql = sql_query.replace('\n', ' ').strip()
        
        # If AI functions are requested, modify the query to add AI function
        if use_ai_functions:
            # Extract the SELECT clause
            select_clause = formatted_sql.split("FROM")[0].strip()
            rest_of_query = "FROM" + formatted_sql.split("FROM", 1)[1]
            
            # Add AI function to explain the results
            ai_function = ', "ai-functions".ai.ai_gen(\'Analyze these customer sales results and explain what insights we can draw from this data\') AS ai_explanation'
            
            # Combine the modified SELECT clause with the rest of the query
            formatted_sql = select_clause + ai_function + " " + rest_of_query
            
            print(f"{Fore.YELLOW}Enhanced query with AI functions:{Style.RESET_ALL}")
            print(textwrap.indent(formatted_sql, "  "))
        
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
    
    headers = results["header"]
    data = results["data"]
    
    # Check if we have AI explanation column
    ai_explanation_idx = None
    for i, header in enumerate(headers):
        if header.lower() == 'ai_explanation':
            ai_explanation_idx = i
            break
    
    # If we have AI explanation, separate it from the data
    ai_explanation = None
    if ai_explanation_idx is not None:
        # Get the first non-empty explanation (they should all be the same)
        for row in data:
            if len(row) > ai_explanation_idx and row[ai_explanation_idx].strip():
                ai_explanation = row[ai_explanation_idx]
                break
        
        # Remove the AI explanation column from headers and data
        headers = headers[:ai_explanation_idx] + headers[ai_explanation_idx+1:]
        data = [row[:ai_explanation_idx] + row[ai_explanation_idx+1:] for row in data]
    
    table = tabulate(data, headers=headers, tablefmt="pretty")
    
    # Add the AI explanation if available
    if ai_explanation:
        table += f"\n\n{Fore.MAGENTA}AI Explanation:{Style.RESET_ALL} {ai_explanation}"
    
    return table

def main():
    parser = argparse.ArgumentParser(description="Run natural language queries against Trino using Trino-AI with AI functions")
    parser.add_argument("query", nargs="?", help="Natural language query to execute")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--ai-functions", "-a", action="store_true", help="Enhance results with AI functions")
    args = parser.parse_args()
    
    if not args.query and not args.interactive:
        parser.print_help()
        sys.exit(1)
    
    def process_query(query, use_ai_functions=False):
        print(f"\n{Fore.GREEN}Query: {query}{Style.RESET_ALL}")
        
        # Convert natural language to SQL
        sql_query, explanation, context = nl_to_sql(query)
        if not sql_query:
            return
        
        # Display the generated SQL
        print(f"\n{Fore.YELLOW}Generated SQL:{Style.RESET_ALL}")
        print(textwrap.indent(sql_query, "  "))
        
        # Execute the SQL query
        results = execute_sql(sql_query, use_ai_functions)
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
        print(f"{Fore.GREEN}=== Trino-AI Natural Language Query Interface with AI Functions ==={Style.RESET_ALL}")
        print(f"{Fore.GREEN}Type your questions in natural language. Type 'exit' or 'quit' to end.{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Use --ai to enable AI function enhancements for a query.{Style.RESET_ALL}")
        
        while True:
            try:
                query = input(f"\n{Fore.CYAN}> {Style.RESET_ALL}")
                if query.lower() in ["exit", "quit", "q"]:
                    break
                
                use_ai = False
                if query.startswith("--ai "):
                    use_ai = True
                    query = query[5:].strip()
                
                if query.strip():
                    process_query(query, use_ai)
            
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            
            except Exception as e:
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    else:
        process_query(args.query, args.ai_functions)

if __name__ == "__main__":
    main() 