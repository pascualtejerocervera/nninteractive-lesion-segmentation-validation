import json
from pathlib import Path
import SimpleITK as sitk


def load_metaio_file(path):
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image)


class Processor:
    def __init__(self, input_dir, output_dir):
        self.__input_dir = Path(input_dir).resolve()
        self.__output_dir = Path(output_dir).resolve()

        self.file_handlers = {
            **{
                k: load_metaio_file
                for k in (".mhd", ".mha", )
            },
        }

        if not self.__input_dir.exists():
            raise ValueError("Input directory does not exist")
        if not self.__output_dir.exists():
            raise ValueError("Output directory does not exist")

    @property
    def input_dir(self):
        return self.__input_dir

    @property
    def output_dir(self):
        return self.__output_dir

    def iterate_inputs(self):
        """
        Iterate over the input directory to process all files.

        Returns
        -------
        An iterator that iterates over all relevant inputs. The result is
        a list of files that should be opened using the defined file handlers.
        """
        extensions = {".mhd", ".mha"}
        for path in self.__input_dir.glob("**"):
            if path.is_file():
                if path.suffix.lower() in extensions:
                    yield (path, )

    def process_item(self, *args):
        raise ValueError("Process item not implemented")

    def process_directory(self):
        processing_results = []
        for input_paths in self.iterate_inputs():
            data_set = {
                "entity": tuple(str(x) for x in input_paths),
                "metrics": None,
                "error_messages": [],
            }
            try:
                result = self.process_item(*data_set)
            except Exception as e:
                data_set["error_messages"] = [str(e)]
            else:
                try:
                    json.dumps(result)
                except ValueError as e:
                    data_set["error_messages"] = \
                        ["Cannot covert result to json: " + str(e)]
                else:
                    data_set["metrics"] = result
        self.format_output(processing_results)

    def format_output(self, data):
        pass