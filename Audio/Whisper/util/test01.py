# Define the URL where Label Studio is accessible and the API key for your user account
LABEL_STUDIO_URL = 'http://localhost:8080'
# API key is available at the Account & Settings > Access Tokens page in Label Studio UI
API_KEY = '8eae80ca770ef06d2eb77f491fb033e3184d1525'

# Import the SDK and the client module
from label_studio_sdk.client import LabelStudio

# Connect to the Label Studio API and check the connection
ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

from label_studio_sdk.label_interface import LabelInterface
from label_studio_sdk.label_interface.create import choices

# Define labeling interface
label_config = LabelInterface.create({
    'text': 'Text',
    'label': choices(['Positive', 'Negative'])
})

# Create a project with the specified title and labeling configuration
project = ls.projects.create(
    title='Text Classification',
    label_config=label_config
)

ls.tasks.create(
    project=project.id,
    data={'text': 'Hello world2'}
)