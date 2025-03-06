import os
import pandas as pd
from tabulate import tabulate
from trino.dbapi import connect
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from logger_config import setup_logger
from env_config import get_trino_credentials

# Set up logger
logger = setup_logger(__name__)

def generate_dataset_report(save_csv=True, save_plots=True):
    """Generate a comprehensive report on all datasets in the registry"""
    try:
        # Get Trino credentials from environment config
        trino_creds = get_trino_credentials()
        
        # Connect to Trino
        conn = connect(
            host=trino_creds['host'],
            port=int(trino_creds['port']),
            user=trino_creds['user'],
            catalog=trino_creds['catalog']
        )
        
        cursor = conn.cursor()
        
        # Query the registry for all dataset information
        cursor.execute("""
        SELECT 
            dataset_id,
            dataset_title,
            category,
            schema_name,
            table_name,
            row_count,
            column_count,
            original_size_bytes,
            parquet_size_bytes,
            etl_timestamp
        FROM nycdata.dataset_registry
        ORDER BY row_count DESC
        """)
        
        columns = ['dataset_id', 'dataset_title', 'category', 
                   'schema_name', 'table_name', 'row_count', 'column_count', 
                   'original_size_bytes', 'parquet_size_bytes', 'etl_timestamp']
        
        data = cursor.fetchall()
        if not data:
            logger.warning("No datasets found in registry")
            return None
            
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        # Add calculated columns
        df['size_mb'] = df['original_size_bytes'] / (1024 * 1024)
        df['parquet_size_mb'] = df['parquet_size_bytes'] / (1024 * 1024)
        df['compression_ratio'] = df['original_size_bytes'] / df['parquet_size_bytes']
        
        # Print summary report
        total_rows = df['row_count'].sum()
        total_original_size = df['original_size_bytes'].sum() / (1024 * 1024 * 1024)  # GB
        total_parquet_size = df['parquet_size_bytes'].sum() / (1024 * 1024 * 1024)  # GB
        avg_compression = df['compression_ratio'].mean()
        
        print("\n===== NYC OPEN DATA DATASET REPORT =====")
        print(f"Total Datasets: {len(df)}")
        print(f"Total Rows: {total_rows:,}")
        print(f"Total Original Size: {total_original_size:.2f} GB")
        print(f"Total Parquet Size: {total_parquet_size:.2f} GB")
        print(f"Average Compression Ratio: {avg_compression:.2f}x")
        print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=======================================\n")
        
        # Print top 10 largest datasets by row count
        print("\nTOP 10 LARGEST DATASETS (BY ROW COUNT):")
        top_by_rows = df.sort_values('row_count', ascending=False).head(10)
        print(tabulate(
            top_by_rows[['dataset_title', 'schema_name', 'table_name', 'row_count', 'size_mb', 'parquet_size_mb']],
            headers=['Dataset', 'Schema', 'Table', 'Rows', 'Size (MB)', 'Parquet (MB)'],
            tablefmt='psql',
            floatfmt=".2f"
        ))
        
        # Print top 10 largest datasets by size
        print("\nTOP 10 LARGEST DATASETS (BY SIZE):")
        top_by_size = df.sort_values('size_mb', ascending=False).head(10)
        print(tabulate(
            top_by_size[['dataset_title', 'schema_name', 'table_name', 'row_count', 'size_mb', 'parquet_size_mb']],
            headers=['Dataset', 'Schema', 'Table', 'Rows', 'Size (MB)', 'Parquet (MB)'],
            tablefmt='psql',
            floatfmt=".2f"
        ))
        
        # Print datasets by category
        print("\nDATASETS BY CATEGORY:")
        category_counts = df.groupby('category').agg({
            'dataset_id': 'count',
            'row_count': 'sum',
            'size_mb': 'sum',
            'parquet_size_mb': 'sum'
        }).sort_values('row_count', ascending=False).reset_index()
        
        print(tabulate(
            category_counts,
            headers=['Category', 'Count', 'Total Rows', 'Size (MB)', 'Parquet (MB)'],
            tablefmt='psql',
            floatfmt=".2f"
        ))
        
        # Save to CSV if requested
        if save_csv:
            os.makedirs('reports', exist_ok=True)
            csv_file = f"reports/dataset_report_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Report saved to {csv_file}")
            
        # Generate plots if requested
        if save_plots:
            os.makedirs('reports', exist_ok=True)
            
            # Set style
            sns.set(style="whitegrid")
            
            # Plot 1: Top 10 datasets by row count
            plt.figure(figsize=(12, 6))
            chart = sns.barplot(data=top_by_rows, x='dataset_title', y='row_count')
            chart.set_xticklabels(chart.get_xticklabels(), rotation=45, ha='right')
            plt.title('Top 10 Datasets by Row Count')
            plt.tight_layout()
            plt.savefig('reports/top_datasets_rows.png')
            
            # Plot 2: Dataset size distribution by category
            plt.figure(figsize=(12, 6))
            chart = sns.barplot(data=category_counts, x='category', y='size_mb')
            chart.set_xticklabels(chart.get_xticklabels(), rotation=45, ha='right')
            plt.title('Total Dataset Size by Category (MB)')
            plt.tight_layout()
            plt.savefig('reports/size_by_category.png')
            
            # Plot 3: Original vs Parquet size
            plt.figure(figsize=(10, 6))
            df_plot = df.sort_values('size_mb', ascending=False).head(15)
            df_plot_melt = pd.melt(df_plot, 
                                  id_vars=['dataset_title'], 
                                  value_vars=['size_mb', 'parquet_size_mb'],
                                  var_name='Format', 
                                  value_name='Size (MB)')
            chart = sns.barplot(data=df_plot_melt, x='dataset_title', y='Size (MB)', hue='Format')
            chart.set_xticklabels(chart.get_xticklabels(), rotation=45, ha='right')
            plt.title('Original vs Parquet Size for Top 15 Datasets')
            plt.tight_layout()
            plt.savefig('reports/original_vs_parquet.png')
            
            logger.info("Generated visualization plots in reports/ directory")
        
        return df
        
    except Exception as e:
        logger.error(f"Error generating dataset report: {str(e)}")
        raise

if __name__ == "__main__":
    generate_dataset_report() 