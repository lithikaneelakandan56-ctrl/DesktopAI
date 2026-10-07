import functools
import subprocess
from typing import Literal

from adbutils import AdbClient
from adbutils.errors import AdbTimeout
from langchain_core.tools import tool as _tool
from langchain_core.tools.structured import StructuredTool
from langchain_tavily import TavilySearch
from PIL import ImageGrab

from src.config import config
from src.ui.overlay import overlay
from src.utils.logger import get_logger

logger = get_logger()

_stop_assistant = False

# Registry to store all tools
_tools_registry = []

adb = AdbClient(host=config.ADB_HOST, port=config.ADB_PORT)


def register_stop_assistant(callback):
    global _stop_assistant
    _stop_assistant = callback


def tool(_func=None, *, return_direct=False):
    """
    A Tool decorator with logger and tool registration.
    """

    def decorator(func):
        @_tool(return_direct=return_direct)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"[{func.__name__}] Called with args: {args}, kwargs: {kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"[{func.__name__}] Result: {result}")
                return result
            except Exception as e:
                logger.error(f"[{func.__name__}] Error: {e}")
                return f"Error in {func.__name__}: {e}"

        # Register the tool
        _tools_registry.append(wrapper)
        return wrapper

    return decorator(_func) if _func is not None else decorator


def adb_required(func):
    """
    A decorator to ensure ADB client is initialized before running the function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            adb.connect(f"{config.MOBILE_HOST}:{config.MOBILE_PORT}", timeout=5)
        except AdbTimeout:
            return "Couldn't connect to mobile."
        devices = adb.list()
        if devices and devices[0].state == "device":
            return func(*args, **kwargs)
        return "Couldn't connect to mobile."

    return wrapper


@tool
def run_command(command: str) -> str:
    """Run a shell command on the local Linux machine."""
    return subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True)


@tool
def open_google_chrome(url: str | None, new_window: bool = False) -> str:
    """
    Open Google Chrome with the specified URL (if url is provided).
    """
    command = f"google-chrome-stable {'--new-window' if new_window else ''} {url if url else ''}"
    logger.info(f"[open_google_chrome] Command: {command}")

    try:
        subprocess.Popen(command, shell=True)
        return f"Google Chrome opened with URL: {url}"
    except Exception as e:
        logger.error(f"[open_google_chrome] Error: {e}")
        return f"Error opening Google Chrome: {e}"


@tool
def open_whatsapp_web() -> str:
    """
    Open WhatsApp Web
    """
    logger.info("[open_whatsapp_web] Opening WhatsApp Web...")
    try:
        subprocess.Popen(["gtk-launch", "chrome-hnpfjngllnobngcgfapefoaidbinmjnm-Default.desktop"])
        return "WhatsApp Web launched."
    except Exception as e:
        logger.error(f"[open_whatsapp_web] Error: {e}")
        return f"Error: {e}"


@tool
def show_popup_widget() -> str:
    """
    Show the popup widget
    """
    overlay.show()
    return "Popup widget opened."


@tool
def hide_popup_widget() -> str:
    """
    Hide the popup widget
    """
    overlay.hide()
    return "Popup widget closed."


@tool
def do_nothing() -> None:
    """
    A function that does nothing
    """
    logger.info("[do_nothing] Called.")


@tool(return_direct=True)
def get_screenshot() -> str:
    """
    Get a screenshot of the current screen
    """
    logger.info("[get_screenshot] Taking screenshot...")
    screenshot = ImageGrab.grab()
    screenshot.save(config.SCREENSHOT_FILE)

    return f"tool_message:get_screenshot:{config.SCREENSHOT_FILE}"


@tool(return_direct=True)
def exit_assistant() -> str:
    """
    Exit / Stop / Shutdown
    """
    if _stop_assistant:
        # Schedule the shutdown to happen after the response is returned
        import threading

        threading.Timer(0.1, _stop_assistant).start()
    else:
        logger.warning("[exit] No stop callback registered.")
        return "No stop callback registered."
    return "Assistant stopped."


@tool
def web_search(
    query: str,
    max_results: int = 5,
    search_depth: Literal["basic", "advanced"] | None = "basic",
) -> str:
    """
    Execute a search query using the Tavily Search API.
    """
    search_tool = TavilySearch(
        api_key=config.TAVILY_API_KEY,
        max_results=max_results,
        search_depth=search_depth,
    )
    try:
        results = search_tool.invoke({"query": query})
        return str(results)
    except Exception as e:
        logger.error(f"[web_search] Error: {e}")
        return f"Error during web search: {e}"


@tool
@adb_required
def mirror_mobile(
    source: Literal["screen", "camera"],
    camera_facing: Literal["front", "back"] | None,
) -> str:
    """
    Mirror the mobile.
    """
    if source == "screen":
        subprocess.Popen("scrcpy", shell=True)
        return "Starting mobile screen mirroring using scrcpy."
    if source == "camera":
        if not camera_facing or camera_facing not in ["front", "back"]:
            camera_facing = "back"
        subprocess.Popen(["scrcpy", "--video-source=camera", f"--camera-facing={camera_facing}"])
        return f"Starting mobile camera {camera_facing} mirroring using scrcpy."
    return "Invalid source. Use 'screen' or 'camera'."


@tool
@adb_required
def get_location() -> str:
    """
    Get the current location from the connected mobile device using ADB.
    """
    device = adb.device()
    return device.shell(r"dumpsys location | grep 'Location\[' | head -n 1 | grep -oE '[0-9]+\.[0-9]+,[0-9]+\.[0-9]+'")


def get_all_tools() -> list[StructuredTool]:
    """
    Get all registered tools.
    """
    return _tools_registry
