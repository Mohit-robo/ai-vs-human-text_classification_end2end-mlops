import subprocess
import sys

def test_streamlit_runs():
    """Check if the Streamlit app boots without crashing"""
    try:
        result = subprocess.run(
            ["streamlit", "run", "main/app.py", "--server.headless", "true"],
            capture_output=True,
            timeout=10
        )
    except subprocess.TimeoutExpired:
        # If it times out, that's fine – app started but kept running
        assert True
        return

    # Should not crash immediately
    assert result.returncode in (0, None)
