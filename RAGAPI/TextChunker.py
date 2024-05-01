from abc import ABC, abstractmethod
import en_core_web_sm
from transformers import AutoTokenizer
import numpy as np
# from mistletoe import Document as MistletoeDocument, span_token, block_token
# from mistletoe.ast_renderer import AstRenderer
# from mistletoe.markdown_renderer import MarkdownRenderer
import os
# from marko import Markdown, ast_renderer

# class Node:
#     def __init__(self, content=None):
#         self.children = []
#         self.content = content

#     def add_child(self, child):
#         self.children.append(child)
#         return self  # Allowing chaining

#     def __repr__(self):
#         return f"{self.__class__.__name__}({self.content!r})"

#     def print_tree(self, level=0):
#         indent = '    ' * level  # Define indentation level
#         print(f"{indent}{self.__class__.__name__}: {self.content}")
#         for child in self.children:
#             child.print_tree(level + 1)

# class Document(Node):
#     pass

# class Section(Node):
#     def __init__(self, content=None, level=1):
#         super().__init__(content)
#         self.level = level

# class Paragraph(Node):
#     pass

# class List(Node):
#     pass

# class ListItem(Node):
#     pass

# class CodeBlock(Node):
#     pass

# class Text(Node):
#     def __init__(self, text):
#         super().__init__(text)


class TextChunker(ABC):
    @abstractmethod
    def chunk_text(text:str, token_chunk_size:int):
        """
        Subclasses must implement this method
        """
        pass  

class ModelTokenizedTextChunker(TextChunker):
    __TEXT_OVERLAP_SENTENCE_LEVEL = -1 # Overlap at a sentence granularity instead of number of characters

    def __init__(self, model_tokenizer_path, text_overlap=__TEXT_OVERLAP_SENTENCE_LEVEL):
        try:
            self.nlp = en_core_web_sm.load()
            self.nlp.add_pipe('sentencizer')
        except Exception as e:
            raise RuntimeError(f"Failed to load Spacy model: {e}")
 
        self.model_tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_path,truncation=True)
        self.text_overlap = int(text_overlap)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
    def chunk_text(self, text:str, token_chunk_size:int=128):
        chunks = []
        current_token_chunk_length = 0
        overlap_text_chunk = ""
        doc = self.nlp(text)
        previous_sentence = ""
        
        # iteratively tokenize each sentence and tally its length and the actual sentence text
        # if the next sentence is going to exceed the chunk size, stop and save it to the chunks array and create an overlap of text
        for sent in doc.sents:
            tokenized_sent = self.model_tokenizer(sent.text, truncation=True, return_tensors="np", padding=False, max_length=token_chunk_size)["input_ids"]        

            if current_token_chunk_length + tokenized_sent.shape[1] <= token_chunk_size:
                # It is safe to continue adding the sentences to our chunk
                overlap_text_chunk += sent.text
                current_token_chunk_length += tokenized_sent.shape[1]
            else:
                # At this point, we're going to exceed the specified max token chunk size, so let's dump the existing buffered
                # chunk and start a new chunk. Add an overlap of text to enable a continue flow of information, in case we truncate paragraphs. 
                chunks.append(overlap_text_chunk)

                if self.text_overlap == self.__TEXT_OVERLAP_SENTENCE_LEVEL:
                    overlap_text_chunk = previous_sentence  # This may break the token_chunk_size rule
                else:    
                    overlap_text_chunk = overlap_text_chunk[-self.text_overlap:] if self.text_overlap < len(overlap_text_chunk) else overlap_text_chunk
                overlap_text_chunk += sent.text
                tokenized_overlap_chunk = self.model_tokenizer(overlap_text_chunk, return_tensors="np", padding=False, truncation=True, max_length=token_chunk_size)["input_ids"] 
                current_token_chunk_length = tokenized_overlap_chunk.shape[1]
            
            previous_sentence = sent.text
        
        # adding the last piece to our text chunk, if it doesn't contain the initial overlap text
        if len(overlap_text_chunk) > 0:
            chunks.append(overlap_text_chunk)    

        return chunks

    # def chunk_text_2(self, text):
    #     chunks = []
    #     current_chunk = ""
    #     current_token_length = 0

    #     for sent in self.nlp(text).sents:
    #         sent_text = sent.text + " "  # Include space to separate sentences.
    #         tokenized_sent = self.model_tokenizer(sent_text, add_special_tokens=False, return_tensors="np")["input_ids"]
    #         sent_token_length = tokenized_sent.shape[1]

    #         if current_token_length + sent_token_length > token_chunk_size:
    #             if current_chunk:  # Ensure not to append empty string initially
    #                 chunks.append(current_chunk.strip())
    #             current_chunk = sent_text
    #             current_token_length = sent_token_length
    #         else:
    #             current_chunk += sent_text
    #             current_token_length += sent_token_length

    #     if current_chunk:  # Add the last chunk if it contains any text.
    #         chunks.append(current_chunk.strip())

    #     return chunks


    

    # def cosine_similarity(self, vec1, vec2):
    #     """Compute cosine similarity between two vectors."""
    #     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # def tokenize_markdown(self, text):
    #     """ Tokenizes markdown text while respecting its structural elements. """
    #     tokens = []
    #     document = Document(text)
    #     for child in document.children:
    #         if isinstance(child, span_token.RawText):
    #             tokens.extend(self.nlp(child.content).sents)
    #         else:
    #             # Treat non-plain text elements as single tokens
    #             tokens.append(type(child).__name__ + child.content)
    #     return tokens
    
    # def token_count(self, text):
    #     tokenized_sent = self.model_tokenizer(text, add_special_tokens=True, return_tensors="np")["input_ids"]
    #     return tokenized_sent.shape[1]
    
    # def chunk_text_semantically(self, text, similarity_threshold=0.75, max_tokens=512):
     
    #     doc = self.nlp(text)
    #     paragraphs = [p for p in text.split('\n') if p.strip()]

    #     chunks = []
    #     current_chunk = []

    #     def get_chunk_text():
    #         return ' '.join(current_chunk)



    #     for paragraph in paragraphs:
    #         if isinstance(paragraph, str):
    #             paragraph_text = paragraph
    #             paragraph_doc = self.nlp(paragraph_text)
    #         else:
    #             # If paragraph is not a string, it's a special block (like code)
    #             paragraph_text = paragraph.text
    #             paragraph_doc = self.nlp(paragraph_text)

    #         if not current_chunk:
    #             current_chunk.append(paragraph_text)
    #             continue

    #         current_chunk_text = get_chunk_text()
    #         current_chunk_doc = self.nlp(current_chunk_text)
    #         similarity = self.cosine_similarity(paragraph_doc.vector, current_chunk_doc.vector)

    #         potential_new_chunk_text = current_chunk_text + ' ' + paragraph_text
    #         if self.token_count(potential_new_chunk_text) <= max_tokens and similarity > similarity_threshold:
    #             current_chunk.append(paragraph_text)
    #         else:
    #             chunks.append(current_chunk_text)
    #             current_chunk = [paragraph_text]
        
    #     if current_chunk:
    #         chunks.append(get_chunk_text())

    #     return chunks


    # def parse_markdown(self, text):
    #     """ Parse the markdown text into structured elements. """
    #     document = Document(text)
    #     sections = []
    #     current_section = []

    #     for token in document.children:
    #         if isinstance(token, block_token.Heading):
    #             if current_section:
    #                 sections.append(current_section)
    #                 current_section = []
    #         current_section.append(token)
    #     if current_section:
    #         sections.append(current_section)
    #     return sections

    # def tokenize_and_chunk(self, text, max_tokens=512):
    #     """ Tokenize text and ensure each chunk does not exceed max_tokens. """
    #     doc = self.nlp(text)
    #     current_chunk = []
    #     current_length = 0
    #     chunks = []

    #     for sent in doc.sents:
    #         sent_text = sent.text.strip()
    #         sent_length = self.token_count(sent_text)

    #         if current_length + sent_length > max_tokens:
    #             chunks.append(" ".join(current_chunk))
    #             current_chunk = [sent_text]
    #             current_length = sent_length
    #         else:
    #             current_chunk.append(sent_text)
    #             current_length += sent_length

    #     if current_chunk:
    #         chunks.append(" ".join(current_chunk))
    #     return chunks

    # def extract_text_from_block(self, block):
    #     """ Recursively extract text from mistletoe block elements """
    #     if hasattr(block, 'children'):
    #         return ' '.join(self.extract_text_from_block(child) for child in block.children)
    #     elif hasattr(block, 'content'):
    #         return block.content
    #     else:
    #         return str(block)


    # def structured_chunk_markdown(self, text):
    #     """ Create structured chunks from Markdown document. """
    #     sections = self.parse_markdown(text)
    #     structured_chunks = []

    #     for section in sections:
    #         for element in section:
    #             text = self.extract_text_from_block(element)
    #             if isinstance(element, block_token.Paragraph):
    #                 chunks = self.tokenize_and_chunk(text)
    #                 structured_chunks.extend(chunks)
    #             else:
    #                 # Treat non-paragraphs as indivisible chunks, but only add if they contain actual text
    #                 if text.strip():
    #                     structured_chunks.append(text)
    #             # if isinstance(element, block_token.Paragraph):
    #             #     text = self.extract_text_from_block(element)
    #             #     chunks = self.tokenize_and_chunk(text)
    #             #     structured_chunks.extend(chunks)
    #             # elif isinstance(element, block_token.CodeFence) or isinstance(element, block_token.List):
    #             #     # Code blocks and lists are treated as single chunks
    #             #     structured_chunks.append(element.content)
    #             # else:
    #             #     # For headers and other elements, directly append
    #             #     structured_chunks.append(element.content)
        
    #     return structured_chunks
    

    # def build_tree_start(self, markdown_text):
    #     doc =  MistletoeDocument(markdown_text)

    #     # parser = Markdown()
    #     # parsed = parser.parse(markdown_text)
    #     # ast = ast_renderer.ASTRenderer().render(parsed)
    #     # import json
    #     # json_output = json.dumps(ast, indent=4)
    #     # print(json_output)
    #     with AstRenderer() as render:
    #         print(render.render(doc))

    #     # with MarkdownRenderer(normalize_whitespace=True) as render:
    #     #     #print(render.render(doc))
    #     #     doc2 = MistletoeDocument(render.render(doc))
    #     #     with AstRenderer() as render:
    #     #         print(render.render(doc2))


    #     return self.build_tree(doc.children)
    
    # def build_tree(self, markdown_element, parent=None):
    #     if parent is None:
    #         parent = Document()

        

    #     for element in markdown_element:
    #         if isinstance(element, block_token.Heading):
    #             new_section = Section(level=element.level, content=self.extract_text_from_block(element))
    #             parent.add_child(new_section)
    #             print("See Heading")
    #             # Recurse into section for nested content
    #             self.build_tree(element.children, new_section)
    #         elif isinstance(element, block_token.Paragraph):
    #             print("See paragraph")
    #             new_paragraph = Paragraph(self.extract_text_from_block(element))
    #             parent.add_child(new_paragraph)
    #         elif isinstance(element, block_token.List):
    #             print("See List")
    #             new_list = List()
    #             for item in element.children:
    #                 new_list.add_child(ListItem(self.extract_text_from_block(item)))
    #             parent.add_child(new_list)
    #         elif isinstance(element, block_token.CodeFence):
    #             print("See Code")
    #             new_code = CodeBlock(element.content)
    #             parent.add_child(new_code)
    #         else:
    #             content=self.extract_text_from_block(element)
    #             print(f"See Unknown: {element} -- {content}")
    #         # handle other types similarly

    #     print("<--return")
    #     return parent

