from Common.Data.Database import RAGDatabase
from Common.OpenAIProviders.OpenAIEmbeddingProvider import OpenAIEmbeddingProvider
from Common.OpenAIProviders.OpenAILLMProvider import OpenAILLMProvider
import json
import os
import argparse
import logging
from flask import Flask, request, jsonify
from cfenv import AppEnv

from TextChunker import ModelTokenizedTextChunker

from RAGDataProvider import RAGDataProvider

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)

RAG_PROVIDER: RAGDataProvider = None

#### Webservice API Routes
@app.route('/upload_files', methods=['POST'])
def upload_file():
    logging.info("Upload starting...")
    try:
        # Get the list of files from the request + optional inputs
        # Token and text chunk size must be specific together
        files = request.files.getlist('files')
        token_chunk_size = request.form.get('token_chunk_size', type=int, default=128)
        topic_display_name = request.form.get('topic_display_name')
        dry_run_level = request.form.get('dry_run_level', type = int, default=0)  # level 0 - no dry run, level 1, dry run with chunk + embedding only , level 2 dry run with chunks only

        if not topic_display_name:
            return jsonify({'error': 'Topic display name is required.'}), 400
        
        if len(files) == 0:
            return jsonify({'error': 'A valid non-blank file is needed.'}), 400
        
        # Set default chunk size if none provided
        if token_chunk_size == None:
            token_chunk_size = 128

        if dry_run_level > 0:
            chunk_data = RAG_PROVIDER.chunk_run(
                                                markdown_files=files, 
                                                topic_display_name=topic_display_name,
                                                token_chunk_size=int(token_chunk_size),
                                                output_embeddings=True if dry_run_level == 1 else False
                                               )
            return jsonify({'chunks_with_embeddings': chunk_data })
        else:
            RAG_PROVIDER.chunk_insert_into_database(
                                            markdown_files=files, 
                                            topic_display_name=topic_display_name,
                                            token_chunk_size=int(token_chunk_size)
                                            )
        return jsonify({'message': 'Embeddings generated for input file and stored successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Create a knowledge base
@app.route('/create_knowledge_base', methods=['POST'])
def create_knowledge_base():
    try:
        topic_display_name = request.form.get('topic_display_name')
        vector_size = request.form.get('vector_size', default=768, type=int)
        domain = request.form.get('topic_domain')
        context_learning_str = request.form.get('context_learning', "[]")

        if not topic_display_name:
            return jsonify({'error': 'Topic display name is required.'}), 400
        
        if not domain:
            return jsonify({'error': 'Topic domain type name is required.'}), 400
        
        if context_learning_str:
            try:
                context_learning = json.loads(context_learning_str)
                is_valid_context_learning = all(
                                    isinstance(item, dict) and 'role' in item and 'content' in item 
                                        for item in context_learning
                                    )

                if not is_valid_context_learning:
                    raise ValueError("Context Learning must be a list of dictionaries with each dictionary having keys, role and content with associated string content")
            except json.JSONDecodeError:
                return jsonify({'error': 'Malformed JSON in context_learning'}), 400

        message = RAG_PROVIDER.create_knowledgebase(topic_display_name=topic_display_name,
                                                    vector_size=vector_size,
                                                    topic_domain=domain,
                                                    context_learning=context_learning
                                                    )
        return jsonify({'message': message})
    except Exception as e:
        output = f"Knowledge base {topic_display_name} creation failed. "
        return jsonify({'error': output + str(e)}), 500

# List all knowledge bases
@app.route('/list_knowledge_bases', methods=['GET'])
def list_knowledge_bases():
    try:
        knowledge_bases = RAG_PROVIDER.get_all_knowledgebases()
        return jsonify({'knowledge_bases': knowledge_bases})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Delete a knowledge base
@app.route('/delete_knowledge_base', methods=['POST'])
def delete_knowledge_base():
    try:
        topic_display_name = request.form.get('topic_display_name')

        if not topic_display_name:
            return jsonify({'error': 'Topic display name is required.'}), 400
        
        RAG_PROVIDER.delete_knowledge_base(topic_display_name=topic_display_name)
        return jsonify({'message': f"Knowledge base {topic_display_name} deleted successfully"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_context_learning', methods=['GET'])
def get_context_learning():
    # Retrieve topic_display_name from query parameters
    topic_display_name = request.args.get('topic_display_name')
    if not topic_display_name:
        return jsonify({'error': 'Topic display name is required.'}), 400

    try:
        context = RAG_PROVIDER.get_knowledge_base_context_learning(topic_display_name)
        if context is None:
            return jsonify({'message': 'No learning context found.'}), 404
        return jsonify(context)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_context_learning', methods=['POST'])
def update_context_learning():

    topic_display_name = request.form.get('topic_display_name')
    context_learning_str = request.form.get('context_learning', None)

    if not topic_display_name:
        return jsonify({'error': 'Topic display name is required.'}), 400
    
    if not context_learning_str:
        return jsonify({'error': 'Learning context is required.'}), 400
    
    context_learning=[]
    try:
        context_learning = json.loads(context_learning_str)

        is_valid_context_learning = all(
                            isinstance(item, dict) and 'role' in item and 'content' in item 
                                for item in context_learning
                            )

        if not is_valid_context_learning:
            raise ValueError("Context Learning must be a list of dictionaries with each dictionary having keys, role and content with associated string content")

    except json.JSONDecodeError:
        return jsonify({'error': 'Malformed JSON in context_learning'}), 400

    try:
        RAG_PROVIDER.update_knowledge_base_context_learning(
                                                    topic_display_name=topic_display_name,
                                                    new_context_learning=context_learning
                                                )
        return jsonify({'message':f"Learning context for {topic_display_name} updated successfully"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Delete all stored embeddings for a given topic
@app.route('/clear_embeddings', methods=['POST'])
def clear_embeddings():
    try:
        topic_display_name = request.form.get('topic_display_name')
        if not topic_display_name:
            return jsonify({'error': 'Topic display name is required.'}), 400
        
        deleted_count = RAG_PROVIDER.clear_knowledgebase_embeddings(topic_display_name=topic_display_name)
        return jsonify({'message': f'All embeddings cleared for {topic_display_name}.', 'deleted_rows': deleted_count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/respond_to_user_query', methods=['POST'])
def respond_to_user_query():
    try:
        query = request.form.get('query')
        schema_table_name = request.form.get('schema_table_name')
        topic_domain = request.form.get('topic_domain')
        do_lost_in_middle_reorder = request.form.get('do_lost_in_middle_reorder', False)
        context_learning_str = request.form.get('context_learning', "[]")

        if not query:
            return jsonify({'error': 'Query is required.'}), 400
        if not topic_domain:
            return jsonify({'error': 'Topic domain is required.'}), 400      
        if not schema_table_name:
            return jsonify({'error': 'Schema + Table name is required.'}), 400        

        if context_learning_str:
            logging.info(context_learning_str)
            try:
                context_learning = json.loads(context_learning_str)

                if context_learning:
                    is_valid_context_learning = all(
                                        isinstance(item, dict) and 'role' in item and 'content' in item 
                                            for item in context_learning
                                        )
                    logging.debug(f"Is Valid json? {is_valid_context_learning}")
                    if not is_valid_context_learning:
                        logging.info("Context Invalid")
                        raise ValueError("Context Learning must be a list of dictionaries with each dictionary having keys, role and content with associated string content")
            except json.JSONDecodeError as e:
                logging.info(e)
                return jsonify({'error': f'Malformed JSON in context_learning: {str(e)}'}), 400


        results = RAG_PROVIDER.respond_to_user_query(
                                                    query=query,
                                                    topic_domain=topic_domain,
                                                    schema_table_name=schema_table_name,
                                                    context_learning=context_learning,
                                                    lost_in_middle_reorder=do_lost_in_middle_reorder
                                                    )
        return jsonify(results)
    except Exception as e:
        logging.info({'error': str(e)})
        return jsonify({'error': str(e)}), 500

def initialize_and_start_service(args):
    """Initialize the application configuration from command line arguments and environment variables."""
    logging.basicConfig(level=logging.INFO)

    cf_env = AppEnv()
    
    # Database configuration
    database_url = args.database if args.database else cf_env.get_service(label='postgres').credentials['jdbcUrl']
    database = RAGDatabase(database_url)

    # API Client configuration
    api_base = args.api_base if args.api_base else cf_env.get_service(label='genai-service').credentials['api_base']
    api_key = args.api_key if args.api_key else cf_env.get_service(label='genai-service').credentials['api_key']
    
    # Model and chunk sizes
    embed_model_name = args.embedding_model if args.embedding_model else os.environ.get("EMBED_MODEL", "hkunlp/instructor-xl")
    is_instructor_model = args.embed_model_is_instructor if args.embed_model_is_instructor else bool(os.environ.get("EMBED_MODEL_IS_INSTRUCTOR", "true").lower()=="true")        

    oaiEmbeddingProvider= OpenAIEmbeddingProvider(
                                            api_base=api_base,
                                            api_key=api_key,
                                            embed_model_name=embed_model_name,
                                            is_instructor_model=is_instructor_model
                                            )
    
    llm_model_name = args.llm_model if args.llm_model else os.environ.get("LLM_MODEL", "Mistral-7B-Instruct-v0.2")
    oai_llm = OpenAILLMProvider(api_base=api_base,
                                api_key=api_key,
                                llm_model_name=llm_model_name,
                                temperature=0.0)
                                
    
    chunker = ModelTokenizedTextChunker(model_tokenizer_path=embed_model_name)
    
    global RAG_PROVIDER
    RAG_PROVIDER = RAGDataProvider(
                                    database=database,
                                    oai_llm=oai_llm,
                                    oai_embed=oaiEmbeddingProvider,
                                    chunker=chunker,
                                    max_results_to_retrieve=20
                                  )


    # Start the Flask application
    app.run(host=args.bind_ip, port=args.port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("-e", "--embedding_model", help="Model name for embeddings")
    parser.add_argument("-m", "--llm_model", help="Model name for embeddings")
    parser.add_argument("-i", "--embed_model_is_instructor", type=bool, help="Model requires instruction")
    parser.add_argument("-s", "--api_base", help="Base URL for the OpenAI API")
    parser.add_argument("-a", "--api_key", help="API key for the OpenAI API")
    parser.add_argument("-d", "--database", help="Database connection string")
    parser.add_argument("-b", "--bind_ip", default="0.0.0.0", help="IP address to bind")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Port to listen on")
    args = parser.parse_args()

    initialize_and_start_service(args)
    