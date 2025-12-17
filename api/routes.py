from flask import Blueprint, jsonify, request
from .bm25_engine import search

api_bp = Blueprint('api', __name__, url_prefix='/api')

PER_PAGE = 10

@api_bp.route('/search', methods=['GET'])
def search_papers():
    query = request.args.get('q', '').strip()
    page = request.args.get('page', 1, type=int)
    
    if not query:
        return jsonify({
            'success': False,
            'message': 'Query parameter q is required',
            'results': []
        }), 400
    
    try:
        all_results = search(query)
        total = len(all_results)
        total_pages = (total + PER_PAGE - 1) // PER_PAGE
        
        start = (page - 1) * PER_PAGE
        end = start + PER_PAGE
        results = all_results[start:end]
        
        return jsonify({
            'success': True,
            'query': query,
            'count': len(results),
            'total': total,
            'page': page,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1,
            'results': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'results': []
        }), 500


@api_bp.route('/paper/<paper_id>', methods=['GET'])
def get_paper(paper_id):
    try:
        from .bm25_engine import RAW_DATA_FILE
        import json
        
        with open(RAW_DATA_FILE, 'r') as f:
            for line in f:
                item = json.loads(line)
                if item.get('id') == paper_id:
                    authors_list = item.get('authors_parsed', [])
                    authors = ', '.join([' '.join(filter(None, a)).strip() for a in authors_list[:10]])
                    if len(authors_list) > 10:
                        authors += f' (+{len(authors_list) - 10} more)'
                    
                    return jsonify({
                        'success': True,
                        'paper': {
                            'paper_id': item.get('id', ''),
                            'title': item.get('title', '').replace('\n', ' '),
                            'authors': authors,
                            'abstract': item.get('abstract', '').replace('\n', ' '),
                            'categories': item.get('categories', ''),
                            'update_date': item.get('update_date', ''),
                            'submitter': item.get('submitter', ''),
                            'journal_ref': item.get('journal-ref', ''),
                            'doi': item.get('doi', ''),
                            'comments': item.get('comments', '')
                        }
                    })
        
        return jsonify({'success': False, 'message': 'Paper not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@api_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})
