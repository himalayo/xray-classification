#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import keras
import sys
import pathlib

is_log_filename = lambda string: string.endswith(".log")
parse_log_filename = lambda filename: filename.stem.split("_")
readeable = lambda component: component.capitalize().replace("_", " ")
component_is_validation = lambda component: component.split("_")[0] == "val"
component_category = lambda component: "Validation" if component_is_validation(component) else "Training"
component_readeable = lambda component: readeable(component) if not component_is_validation(component) else readeable("_".join(component.split("_")[1:]))

stylized_model_names = {
    'convnext': 'ConvNeXt',
    "efficientnet": 'EfficientNet',
    "mobilenet": "MobileNet",
    "vgg16": "VGG16",
    "inception": "InceptionV3",
    "resnet50": "ResNet50"
}

def plot_component(filename, data, component_name, lr=1e-5):
    attributes = parse_log_filename(filename)
    if len(attributes) == 2:
        model_name, dense_units = attributes
    if len(attributes) == 3:
        model_name, dense_units, lr = attributes

    if model_name in stylized_model_names:
        model_name = stylized_model_names[model_name]

    plt.plot(data[component_name], label=f"{model_name} {dense_units} {lr}")

def plot_result(results, component):
    fig = plt.figure()
    for data, filename in results:
        plot_component(filename, data, component)
    plt.title(f"{component_category(component)} {component_readeable(component)}")
    plt.ylabel(component_readeable(component))
    plt.xlabel("Epochs")
    plt.legend()
    fig.savefig(f"output/{component}.png")

def plot_results(results, component):
    plot_result(results, component)
    plot_result(results, f"val_{component}")


if __name__ == "__main__":
    filenames = map(pathlib.Path, filter(is_log_filename, sys.argv))
    results = list(map(lambda filename: (pd.read_csv(filename), filename), filenames))
    all_columns = results[0][0].columns.tolist()
    selected_columns = all_columns if "--all" in sys.argv else list(filter(lambda x: x in all_columns, sys.argv))
    for column in selected_columns:
        plot_result(results, column)
    plt.show()
