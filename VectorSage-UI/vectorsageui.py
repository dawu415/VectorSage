from urllib.parse import urljoin
from typing import List, Dict, Any
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import gradio
import  json
import logging
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)

@dataclass_json
@dataclass
class KnowledgeBase:
    topic_display_name: str
    schema_table_name: str
    topic_domain: str
    context_learning: List[Dict[str, Any]]
    id: int = None

@dataclass
class VectorSageUI:
    llm_rag_services_host: str
    listen_port: int
    cached_knowledgebases: List[KnowledgeBase] = field(default_factory=list)
    current_knowledgebase: KnowledgeBase = None

    ### GRADIO TOP LEVEL FUNCTIONS
    def _init_history(self, messages_history: gradio.State):
        messages_history = []
        return messages_history

    def _process_user_input(self, user_query: gradio.Textbox, history: gradio.Chatbot):
        return "", history + [[user_query, None]]

    def _complete_chat(self, history: gradio.Chatbot, messages_history: gradio.State):
        user_query = history[-1][0]
        ## No History for now. 
        # messages = messages_history
        if self.current_knowledgebase:
            cur_kb = self.current_knowledgebase
            endpoint = urljoin(self.llm_rag_services_host, "respond_to_user_query")
            response = requests.post(endpoint,
                                    data = {
                                        "query": user_query,
                                        "topic_domain": cur_kb.topic_domain,
                                        "schema_table_name": cur_kb.schema_table_name,
                                        "do_lost_in_middle_reorder": True,
                                        "context_learning": json.dumps(cur_kb.context_learning)
                                    })
        
        logging.info(f"Response Content: \n {response}")
        logging.info(f"Response Content: \n {response.json().strip()}")
        
        # Update the chat history with the LLM response and return
        messages_history += [{"role": "user", "content": user_query}]
        messages_history += [{"role": "assistant", "content": response.json().strip()}]
        history[-1][1] = response.json().strip()

        return history, messages_history
        
    # Get the list of Knowledge Bases available
    def _fetch_dropdown_knowledge_options(self):
        """Fetch options for the dropdown from the database."""
        endpoint = urljoin(self.llm_rag_services_host,"list_knowledge_bases")
        response = requests.get(endpoint)
        knowledge_bases_json = response.json()['knowledge_bases']
        knowledge_bases = [KnowledgeBase.from_json(json.dumps(kb)) for kb in knowledge_bases_json]
        return knowledge_bases

    def _handle_dropdown_selection(self, selected_topic: str):
        """Handle the dropdown selection and fetch more data."""
        logging.info(f"handling dropdown - selected: {selected_topic}")
        kb_list = self.cached_knowledgebases
        foundItem = next(filter(lambda x: x.topic_display_name == selected_topic, kb_list), kb_list[0])
        self.current_knowledgebase = foundItem

    def _refresh_dropdown_data(self):
        """Function to refresh dropdown data from the database."""
        kb_list = self._fetch_dropdown_knowledge_options()
        self.cached_knowledgebases = kb_list
        cur_kb = self.current_knowledgebase
        
        topic_display_name = None
        display_options = []
        if cur_kb == None and len(kb_list) > 0:
            self.current_knowledgebase = kb_list[0]
            display_options = [kb.topic_display_name for kb in kb_list]
            topic_display_name = self.current_knowledgebase.topic_display_name
        elif cur_kb != None and len(kb_list) > 0:
            foundItem = next(filter(lambda x: x.topic_display_name == cur_kb.topic_display_name, kb_list), kb_list[0])
            display_options = [kb.topic_display_name for kb in kb_list]
            self.current_knowledgebase = foundItem
            topic_display_name = foundItem.topic_display_name
        
        return gradio.Dropdown(choices=display_options, label="Knowledge Base", value = topic_display_name, interactive=True)

    def start(self):
        with gradio.Blocks(fill_height=True) as grai_ui:
            gradio.Markdown("""<h1><center>VectorSage - GenAI on TAS</center></h1>""")
            with gradio.Accordion("Configure AI",open=False) as scene_accordion:
                with gradio.Row():
                    kb_dropdown = self._refresh_dropdown_data()
                    refresh_button = gradio.Button("Refresh List")

            kb_dropdown.change(self._handle_dropdown_selection, inputs=[kb_dropdown], outputs=[], preprocess=False, postprocess=False)
            refresh_button.click(self._refresh_dropdown_data, inputs=[], outputs=kb_dropdown)
            
            chatbot = gradio.Chatbot(scale=2)
            message_textbox = gradio.Textbox(placeholder="Enter Your Query Here")
            clear_button = gradio.Button("Clear Session")
            message_history = gradio.State([])

            # UI Init
            grai_ui.load( 
                            lambda: None, None, chatbot, queue=False
                        ).success(
                                self._init_history, [message_history], [message_history]
                                 )

            # Chained actions on submission
            clear_button.click(
                            lambda: None, None, chatbot, queue=False
                            ).success(
                                        self._init_history, [message_history], [message_history]
                                    )
                        
            message_textbox.submit(
                                    self._process_user_input, [message_textbox, chatbot], [message_textbox, chatbot], queue=False
                                ).success(
                                            self._complete_chat, [chatbot, message_history], [chatbot, message_history],
                                        )

        grai_ui.launch(server_name="0.0.0.0", server_port = self.listen_port, debug = True)    


