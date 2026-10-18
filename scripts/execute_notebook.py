"""Execute the notebook and persist outputs, including a useful failed state."""
from pathlib import Path
import argparse
import os
import sys
import traceback
import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager

root = Path(__file__).resolve().parents[1]
path = root / "Baseline_vs_Reranking.ipynb"
notebook = nbformat.read(path, as_version=4)
os.environ.setdefault("IPYTHONDIR", str(root / ".cache" / "ipython"))
os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
parser = argparse.ArgumentParser()
parser.add_argument("--in-process", action="store_true",
                    help="Execute real IPython cells without a kernel server (for environments that prohibit local sockets).")
args = parser.parse_args()
if args.in_process:
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    import matplotlib
    matplotlib.use("module://matplotlib_inline.backend_inline")
    from matplotlib_inline.backend_inline import configure_inline_support
    shell = InteractiveShell.instance()
    configure_inline_support(shell, "module://matplotlib_inline.backend_inline")
    notebook.metadata["execution_method"] = "Real IPython execution in process; no kernel server."
    count = 0
    try:
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            count += 1
            print(f"Executing cell {i + 1}/{len(notebook.cells)}", flush=True)
            cell.execution_count = count
            with capture_output() as captured:
                outcome = shell.run_cell(cell.source, store_history=False)
            cell.outputs = []
            if captured.stdout:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
            if captured.stderr:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
            for output in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
            error = outcome.error_before_exec or outcome.error_in_exec
            if error:
                cell.outputs.append(nbformat.v4.new_output("error", ename=type(error).__name__,
                    evalue=str(error), traceback=traceback.format_exception(error)))
                raise error
    finally:
        nbformat.write(notebook, path)
    print(f"Executed and saved {path.name} with real IPython cell outputs.")
    sys.exit(0)

manager = AsyncKernelManager(kernel_name="python3")
manager.kernel_spec.argv[0] = sys.executable
client = NotebookClient(notebook, km=manager, timeout=1800, startup_timeout=30,
                        kernel_name="python3", resources={"metadata": {"path": str(root)}},
                        on_cell_start=lambda cell, cell_index: print(f"Executing cell {cell_index + 1}/{len(notebook.cells)}", flush=True))
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
print(f"Executed and saved {path.name}.")
