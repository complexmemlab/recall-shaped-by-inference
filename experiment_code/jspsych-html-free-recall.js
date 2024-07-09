var htmlFreeRecall = (function (jspsych) {
  "use strict";

  const info = {
    name: "html-free-recall",
    parameters: {
      stimulus: {
        type: jspsych.ParameterType.HTML_STRING,
        pretty_name: "Stimulus",
        default: undefined,
      },
      choices: {
        type: jspsych.ParameterType.KEYS,
        pretty_name: "Choices",
        default: "ALL_KEYS",
      },
      prompt: {
        type: jspsych.ParameterType.HTML_STRING,
        pretty_name: "Prompt",
        default: null,
      },
      stimulus_duration: {
        type: jspsych.ParameterType.INT,
        pretty_name: "Stimulus duration",
        default: null,
      },
      trial_duration: {
        type: jspsych.ParameterType.INT,
        pretty_name: "Trial duration",
        default: null,
      },
      response_ends_trial: {
        type: jspsych.ParameterType.BOOL,
        pretty_name: "Response ends trial",
        default: true,
      },
    },
  };

  class HtmlFreeRecallPlugin {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }

    trial(display_element, trial) {
      var new_html =
        '<div id="jspsych-html-keyboard-response-stimulus">' +
        trial.stimulus +
        "</div>";

      if (trial.prompt !== null) {
        new_html += trial.prompt;
      }

      display_element.innerHTML = new_html;

      var response = {
        keys: [],
        rt: [],
      };

      const end_trial = () => {
        this.jsPsych.pluginAPI.clearAllTimeouts();

        if (typeof keyboardListener !== "undefined") {
          this.jsPsych.pluginAPI.cancelKeyboardResponse(keyboardListener);
        }

        // Filter out "Enter" keys and join the remaining keys into a single string
        const combinedKeys = response.keys
          .filter((key) => key !== "enter")
          .join("");

        var trial_data = {
          rt: response.rt,
          stimulus: trial.stimulus,
          response: combinedKeys,
        };

        display_element.innerHTML = "";

        this.jsPsych.finishTrial(trial_data);
      };

      var after_response = (info) => {
        response.keys.push(info.key);
        response.rt.push(info.rt);

        if (info.key === "enter") {
          if (trial.response_ends_trial) {
            end_trial();
          }
        }
      };

      if (trial.choices != "NO_KEYS") {
        var keyboardListener = this.jsPsych.pluginAPI.getKeyboardResponse({
          callback_function: after_response,
          valid_responses: trial.choices,
          rt_method: "performance",
          persist: true,
          allow_held_key: false,
        });
      }

      if (trial.stimulus_duration !== null) {
        this.jsPsych.pluginAPI.setTimeout(() => {
          display_element.querySelector(
            "#jspsych-html-keyboard-response-stimulus",
          ).style.visibility = "hidden";
        }, trial.stimulus_duration);
      }

      if (trial.trial_duration !== null) {
        this.jsPsych.pluginAPI.setTimeout(end_trial, trial.trial_duration);
      }
    }
  }
  HtmlFreeRecallPlugin.info = info;

  return HtmlFreeRecallPlugin;
})(jsPsychModule);
