import os
from config import Config
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.components.fetchers.link_content import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack import Pipeline
import gradio as gr

# Setting up the environment variable for Hugging Face API authentication
os.environ["HF_API_TOKEN"] = Config.HUGGINGFACE_API_KEY
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

def define_prompt_template():
    # Defines the template used for generating prompts in the pipeline
    prompt_template = """
    According to these documents:

    {% for doc in documents %}
      {{ doc.content }}
    {% endfor %}

    Answer the given question: {{question}}
    Answer:
    """
    prompt_builder = PromptBuilder(template=prompt_template)
    
    return prompt_builder
    

# Create the RAG pipeline
def create_pipeline():
    # Initialize the language model generator from Hugging Face
    generator = HuggingFaceAPIGenerator(
        api_type="serverless_inference_api",
        api_params={"model": LLM_MODEL}
    )

    # Define the pipeline components for fetching and processing documents
    fetcher = LinkContentFetcher()
    converter = HTMLToDocument()
    document_splitter = DocumentSplitter(split_by="word", split_length=50)
    similarity_ranker = TransformersSimilarityRanker(top_k=3)
    
    # Define the prompt builder using a custom template
    prompt_builder = define_prompt_template()
    
    # Create a pipeline and add components
    pipeline = Pipeline()
    pipeline.add_component("fetcher", fetcher)
    pipeline.add_component("converter", converter)
    pipeline.add_component("splitter", document_splitter)
    pipeline.add_component("ranker", similarity_ranker)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", generator)

    # Connect the components to define the flow of data through the pipeline
    pipeline.connect("fetcher.streams", "converter.sources")
    pipeline.connect("converter.documents", "splitter.documents")
    pipeline.connect("splitter.documents", "ranker.documents")
    pipeline.connect("ranker.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.prompt")

    return pipeline
    
def ask(question, url):
    # Function to process a question and a URL and return the answer using the defined pipeline
    pipeline = create_pipeline()
    result = pipeline.run({
        "prompt_builder": {"question": question},
        "ranker": {"query": question},
        "fetcher": {"urls": [url]},
        "llm": {"generation_kwargs": {"max_new_tokens": 350}}
    })
    return result['llm']['replies'][0]


def setup_gradio_interface(ask_function):
    # Sets up and returns a Gradio interface using the specified function for processing user inputs
    iface = gr.Interface(
        fn=ask_function, 
        inputs=[gr.Textbox(label="Question"), gr.Textbox(label="URL")], 
        outputs="text",
        title="RAG Pipeline with Hugging Face and Gradio",
        description="Ask a question about the content of a specific URL.",
        examples=[
            ["What is computer vision?", "https://haystack.deepset.ai/blog/introducing-haystack-2-beta-and-advent"],
            ["What is Natural Language Processing?", "https://en.wikipedia.org/wiki/Natural_language_processing"]
        ],
        theme=gr.themes.Soft(),
        allow_flagging="never"
    )
    return iface
    
# Main execution block to initialize and launch the Gradio interface
if __name__ == "__main__":
    iface = setup_gradio_interface(ask)
    iface.launch(debug=True)  # Launch the interface with debugging enabled
