import sys

from dotenv import load_dotenv

load_dotenv()

from builder_agent.cli import main  # noqa: E402

sys.exit(main())
