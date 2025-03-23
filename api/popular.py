from data_loader.data_loader import DataLoader

@bp.route('/load', methods=['POST'])
def load_popular_datasets():
    """Load popular datasets into Trino"""
    try:
        count = request.json.get('count', 5)
        domain = request.json.get('domain', 'data.cityofnewyork.us')
        
        # Discover popular datasets
        discovery = SocrataDiscovery()
        datasets = discovery.find_popular_datasets(domain=domain, limit=count)
        
        # Create data loader for Trino ETL
        loader = DataLoader()
        
        # Process datasets asynchronously
        def process_datasets():
            results = []
            for dataset in datasets:
                dataset_id = dataset.get('dataset_id')
                dataset_name = dataset.get('name', 'Unknown')
                current_app.logger.info(f"Loading dataset {dataset_id}: {dataset_name}")
                
                # Use DataLoader instead of deprecated method
                result = loader.load_dataset(dataset_id)
                
                results.append({
                    'dataset_id': dataset_id,
                    'name': dataset_name,
                    'success': result['success'],
                    'errors': result.get('errors', []),
                    'table_name': result.get('table_name')
                })
                
            return results
        
        # Start background task
        thread = Thread(target=process_datasets)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': f'Loading {len(datasets)} datasets in the background',
            'datasets': [{'id': d.get('dataset_id'), 'name': d.get('name')} for d in datasets]
        }), 202
        
    except Exception as e:
        current_app.logger.error(f"Error loading popular datasets: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500 