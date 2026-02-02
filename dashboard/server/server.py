"""
ML Project Dashboard Backend Server

A lightweight Flask server that provides read-only API endpoints
for the dashboard to access project data.

Run this server from the dashboard/server directory to avoid
import conflicts with the local flask/ directory in the project.
"""

import os
import re
import sys
from pathlib import Path

# Get the project root directory (two levels up from server/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DASHBOARD_ROOT = Path(__file__).parent.parent

# Remove project root from path to avoid flask/ directory conflict
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
if '' in sys.path:
    sys.path.remove('')

# Now we can safely import Flask
from flask import Flask, jsonify, request, send_from_directory, make_response
import pandas as pd
import yaml

app = Flask(__name__, static_folder=str(DASHBOARD_ROOT))


# ============================================================================
# CORS Middleware (manual implementation to avoid flask-cors import issues)
# ============================================================================

@app.after_request
def add_cors_headers(response):
    """Add CORS headers to all responses for local development."""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


@app.before_request
def handle_preflight():
    """Handle CORS preflight requests."""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response


# ============================================================================
# Static File Serving
# ============================================================================

@app.route('/')
def serve_index():
    """Serve the main dashboard HTML file."""
    return send_from_directory(str(DASHBOARD_ROOT), 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS, etc.)."""
    # Don't serve API routes as static files
    if path.startswith('api/'):
        return jsonify({'error': True, 'message': 'Not found'}), 404
    return send_from_directory(str(DASHBOARD_ROOT), path)


# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/api/dataset')
def get_dataset():
    """
    Return paginated dataset from data/diabetes.csv.
    
    Query Parameters:
        page (int): Page number (1-indexed, default: 1)
        pageSize (int): Number of rows per page (default: 100)
    
    Returns:
        JSON with columns, data, total count, and pagination info
    """
    try:
        csv_path = PROJECT_ROOT / 'data' / 'diabetes.csv'
        
        if not csv_path.exists():
            return jsonify({
                'error': True,
                'message': f'Dataset file not found: {csv_path}',
                'code': 'FILE_NOT_FOUND'
            }), 404
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('pageSize', 100, type=int)
        
        # Validate parameters
        if page < 1:
            return jsonify({
                'error': True,
                'message': 'Page number must be >= 1',
                'code': 'INVALID_PARAMETER'
            }), 400
        
        if page_size < 1 or page_size > 1000:
            return jsonify({
                'error': True,
                'message': 'Page size must be between 1 and 1000',
                'code': 'INVALID_PARAMETER'
            }), 400
        
        # Calculate pagination
        total_rows = len(df)
        total_pages = (total_rows + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Get paginated data
        page_data = df.iloc[start_idx:end_idx]
        
        return jsonify({
            'columns': df.columns.tolist(),
            'data': page_data.to_dict(orient='records'),
            'total': total_rows,
            'page': page,
            'pageSize': page_size,
            'totalPages': total_pages
        })
        
    except Exception as e:
        return jsonify({
            'error': True,
            'message': str(e),
            'code': 'PARSE_ERROR'
        }), 500


@app.route('/api/dvc-info')
def get_dvc_info():
    """
    Return DVC versioning information from data/diabetes.csv.dvc.
    
    Returns:
        JSON with md5 hash, size, and path
    """
    try:
        dvc_path = PROJECT_ROOT / 'data' / 'diabetes.csv.dvc'
        
        if not dvc_path.exists():
            return jsonify({
                'error': True,
                'message': f'DVC file not found: {dvc_path}',
                'code': 'FILE_NOT_FOUND'
            }), 404
        
        # Parse the YAML file
        with open(dvc_path, 'r') as f:
            dvc_data = yaml.safe_load(f)
        
        # Extract the output information
        if 'outs' in dvc_data and len(dvc_data['outs']) > 0:
            out = dvc_data['outs'][0]
            return jsonify({
                'md5': out.get('md5', ''),
                'size': out.get('size', 0),
                'path': out.get('path', ''),
                'hash': out.get('hash', 'md5')
            })
        else:
            return jsonify({
                'error': True,
                'message': 'No outputs found in DVC file',
                'code': 'PARSE_ERROR'
            }), 500
            
    except yaml.YAMLError as e:
        return jsonify({
            'error': True,
            'message': f'YAML parse error: {str(e)}',
            'code': 'PARSE_ERROR'
        }), 500
    except Exception as e:
        return jsonify({
            'error': True,
            'message': str(e),
            'code': 'INTERNAL_ERROR'
        }), 500


@app.route('/api/feature-store/config')
def get_feature_store_config():
    """
    Return Feast feature store configuration.
    
    Returns:
        JSON with project name, provider, and online store config
    """
    try:
        config_path = PROJECT_ROOT / 'feature_store' / 'feature_repo' / 'feature_store.yaml'
        
        if not config_path.exists():
            return jsonify({
                'error': True,
                'message': f'Feature store config not found: {config_path}',
                'code': 'FILE_NOT_FOUND'
            }), 404
        
        # Parse the YAML file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return jsonify({
            'project': config.get('project', ''),
            'provider': config.get('provider', ''),
            'registry': config.get('registry', ''),
            'onlineStore': config.get('online_store', {})
        })
        
    except yaml.YAMLError as e:
        return jsonify({
            'error': True,
            'message': f'YAML parse error: {str(e)}',
            'code': 'PARSE_ERROR'
        }), 500
    except Exception as e:
        return jsonify({
            'error': True,
            'message': str(e),
            'code': 'INTERNAL_ERROR'
        }), 500


@app.route('/api/feature-store/views')
def get_feature_store_views():
    """
    Return feature store entity and feature view definitions.
    
    Parses the example_repo.py file using regex to extract
    entity definitions and feature view schemas.
    
    Returns:
        JSON with entities and featureViews arrays
    """
    try:
        repo_path = PROJECT_ROOT / 'feature_store' / 'feature_repo' / 'example_repo.py'
        
        if not repo_path.exists():
            return jsonify({
                'error': True,
                'message': f'Feature repo file not found: {repo_path}',
                'code': 'FILE_NOT_FOUND'
            }), 404
        
        with open(repo_path, 'r') as f:
            content = f.read()
        
        # Parse entities
        entities = parse_entities(content)
        
        # Parse feature views
        feature_views = parse_feature_views(content)
        
        return jsonify({
            'entities': entities,
            'featureViews': feature_views
        })
        
    except Exception as e:
        return jsonify({
            'error': True,
            'message': str(e),
            'code': 'PARSE_ERROR'
        }), 500


@app.route('/api/feature-store/data')
def get_feature_store_data():
    """
    Return feature store data from parquet files.
    
    Query Parameters:
        page (int): Page number (1-indexed, default: 1)
        pageSize (int): Number of rows per page (default: 50)
    
    Returns:
        JSON with predictors and targets data, including entity keys and timestamps
    """
    try:
        predictor_path = PROJECT_ROOT / 'data' / 'artifacts' / 'predictor.parquet'
        target_path = PROJECT_ROOT / 'data' / 'artifacts' / 'target.parquet'
        
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('pageSize', 50, type=int)
        
        # Validate parameters
        if page < 1:
            return jsonify({
                'error': True,
                'message': 'Page number must be >= 1',
                'code': 'INVALID_PARAMETER'
            }), 400
        
        if page_size < 1 or page_size > 500:
            return jsonify({
                'error': True,
                'message': 'Page size must be between 1 and 500',
                'code': 'INVALID_PARAMETER'
            }), 400
        
        result = {
            'predictors': None,
            'targets': None,
            'ttl': 86400 * 2  # 2 days in seconds as defined in example_repo.py
        }
        
        # Load predictor data
        if predictor_path.exists():
            df_pred = pd.read_parquet(predictor_path)
            total_pred = len(df_pred)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            page_data = df_pred.iloc[start_idx:end_idx]
            
            # Convert timestamps to ISO format strings for JSON serialization
            data_records = page_data.to_dict(orient='records')
            for record in data_records:
                for key, value in record.items():
                    if hasattr(value, 'isoformat'):
                        record[key] = value.isoformat()
            
            result['predictors'] = {
                'columns': df_pred.columns.tolist(),
                'data': data_records,
                'total': total_pred,
                'page': page,
                'pageSize': page_size,
                'totalPages': (total_pred + page_size - 1) // page_size
            }
        
        # Load target data
        if target_path.exists():
            df_target = pd.read_parquet(target_path)
            total_target = len(df_target)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            page_data = df_target.iloc[start_idx:end_idx]
            
            # Convert timestamps to ISO format strings for JSON serialization
            data_records = page_data.to_dict(orient='records')
            for record in data_records:
                for key, value in record.items():
                    if hasattr(value, 'isoformat'):
                        record[key] = value.isoformat()
            
            result['targets'] = {
                'columns': df_target.columns.tolist(),
                'data': data_records,
                'total': total_target,
                'page': page,
                'pageSize': page_size,
                'totalPages': (total_target + page_size - 1) // page_size
            }
        
        if result['predictors'] is None and result['targets'] is None:
            return jsonify({
                'error': True,
                'message': 'No feature store data files found',
                'code': 'FILE_NOT_FOUND'
            }), 404
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': True,
            'message': str(e),
            'code': 'INTERNAL_ERROR'
        }), 500


def parse_entities(content: str) -> list:
    """
    Parse Entity definitions from Python code.
    
    Args:
        content: Python source code as string
        
    Returns:
        List of entity dictionaries with name, valueType, description
    """
    entities = []
    
    # Pattern to match Entity definitions
    entity_pattern = r'Entity\s*\(\s*name\s*=\s*["\']([^"\']+)["\']\s*,\s*value_type\s*=\s*ValueType\.(\w+)(?:\s*,\s*description\s*=\s*["\']([^"\']+)["\'])?'
    
    matches = re.finditer(entity_pattern, content, re.DOTALL)
    
    for match in matches:
        entities.append({
            'name': match.group(1),
            'valueType': match.group(2),
            'description': match.group(3) if match.group(3) else ''
        })
    
    return entities


def parse_feature_views(content: str) -> list:
    """
    Parse FeatureView definitions from Python code.
    
    Args:
        content: Python source code as string
        
    Returns:
        List of feature view dictionaries with name, ttl, entities, schema, etc.
    """
    feature_views = []
    
    # Pattern to match FeatureView definitions
    # This is a simplified pattern that captures the main components
    fv_pattern = r'FeatureView\s*\(\s*name\s*=\s*["\']([^"\']+)["\']'
    
    # Find all FeatureView blocks
    fv_blocks = re.split(r'(?=\w+\s*=\s*FeatureView\s*\()', content)
    
    for block in fv_blocks:
        if 'FeatureView(' not in block:
            continue
            
        fv = {}
        
        # Extract name
        name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', block)
        if name_match:
            fv['name'] = name_match.group(1)
        else:
            continue
        
        # Extract TTL
        ttl_match = re.search(r'ttl\s*=\s*timedelta\s*\(\s*seconds\s*=\s*([^)]+)\)', block)
        if ttl_match:
            ttl_expr = ttl_match.group(1).strip()
            # Evaluate simple expressions like 86400*2
            try:
                fv['ttl'] = eval(ttl_expr)
            except:
                fv['ttl'] = ttl_expr
        
        # Extract entities
        entities_match = re.search(r'entities\s*=\s*\[([^\]]+)\]', block)
        if entities_match:
            entity_refs = entities_match.group(1).strip()
            # Extract entity variable names
            fv['entities'] = [e.strip() for e in entity_refs.split(',') if e.strip()]
        
        # Extract schema fields
        schema_match = re.search(r'schema\s*=\s*\[(.*?)\]', block, re.DOTALL)
        if schema_match:
            schema_content = schema_match.group(1)
            fields = []
            
            # Pattern to match Field definitions
            field_pattern = r"Field\s*\(\s*name\s*=\s*['\"]([^'\"]+)['\"]\s*,\s*dtype\s*=\s*(\w+)"
            field_matches = re.finditer(field_pattern, schema_content)
            
            for fm in field_matches:
                fields.append({
                    'name': fm.group(1),
                    'dtype': fm.group(2)
                })
            
            fv['schema'] = fields
        
        # Extract online flag
        online_match = re.search(r'online\s*=\s*(True|False)', block)
        if online_match:
            fv['online'] = online_match.group(1) == 'True'
        
        feature_views.append(fv)
    
    return feature_views


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    args = parser.parse_args()
    
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dashboard root: {DASHBOARD_ROOT}")
    print("Starting ML Project Dashboard server...")
    print(f"Dashboard available at: http://localhost:{args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=True)
