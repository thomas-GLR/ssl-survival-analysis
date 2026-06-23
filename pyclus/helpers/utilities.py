import tempfile
import os
import logging
import subprocess
from typing import Dict, Any, Callable


logger = logging.getLogger(__name__)


class Utilities:
    JAVA_ADD_OPENS = "--add-opens java.base/java.util=ALL-UNNAMED"

    @staticmethod
    def find_an_available_root_temp_dir():
        candidate = tempfile.gettempdir()
        # test if writeable
        test_file = os.path.join(candidate, "test.txt")
        try:
            with open(test_file, "w") as f:
                print("?", file=f)
        except:
            logger.error(f"The temp directory {candidate} is not writable. You won't be able to do much.")
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
        return candidate

    @staticmethod
    def determine_java_command():
        java_opens = [Utilities.JAVA_ADD_OPENS, ""]
        for java_option in java_opens:
            command = [part for part in f"java {java_option} -version".split(" ") if part]
            exit_code = subprocess.call(
                command,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL
            )
            if exit_code == 0:
                return java_option
        raise RuntimeError(
            "Cannot run java properly. Please, install it or add it to path, "
            "so that the command '> java -version' executes properly in the command line."
        )

    @staticmethod
    def get_list_of_allowed_values(
            variables: Dict[str, Any],
            name_filter: Callable[[str], bool]
    ):
        return [value for name, value in variables.items() if name_filter(name)]


if __name__ == "__main__":
    Utilities.find_an_available_root_temp_dir()
    Utilities.determine_java_command()
