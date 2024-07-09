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
      button_string: {
        type: jspsych.ParameterType.HTML_STRING,
        pretty_name: "Button HTML",
        default: null,
      },
      button_delay: {
        type: jspsych.ParameterType.INT,
        pretty_name: "Button delay",
        default: 0,
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

      // Add a textbox for input
      new_html +=
        '<div><input type="text" id="jspsych-html-keyboard-response-textbox" autocomplete="off" /></div>';

      if (trial.prompt !== null) {
        new_html += trial.prompt;
      }

      if (trial.button_string !== null) {
        new_html +=
          '<div><button id="jspsych-html-keyboard-response-button" style="display: none; justify-content:center">' +
          trial.button_string +
          "</button></div>";
      }

      display_element.innerHTML = new_html;

      // Focus the textbox to enable typing immediately
      var textbox = display_element.querySelector(
        "#jspsych-html-keyboard-response-textbox",
      );
      textbox.focus();
      var response = {
        rt: null,
        button: null,
      };
      // Add an event listener to the button, if it exists, and show it after a delay
      if (trial.button_string !== null) {
        var button = display_element.querySelector(
          "#jspsych-html-keyboard-response-button",
        );
        button.addEventListener("click", () => {
          response.button = "pressed"; // Set button to "pressed" when clicked
          if (trial.response_ends_trial) {
            end_trial();
          }
        });

        this.jsPsych.pluginAPI.setTimeout(() => {
          button.style.display = "initial"; // Show the button after the delay
        }, trial.button_delay);
      }
      const end_trial = () => {
        this.jsPsych.pluginAPI.clearAllTimeouts();

        textbox.removeEventListener("keydown", checkForEnterKey); // Remove the event listener

        var trial_data = {
          rt: response.rt,
          stimulus: trial.stimulus,
          response: textbox.value, // Get the value of the textbox
          button: response.button,
        };

        display_element.innerHTML = "";

        this.jsPsych.finishTrial(trial_data);
      };

      // Function to check for space key press in the textbox
      const checkForEnterKey = (event) => {
        if (
          textbox.value.trim() !== "" &&
          (event.key === "Space" || event.key === " ")
        ) {
          response.rt = performance.now() - start_time;
          if (trial.response_ends_trial) {
            end_trial();
          }
        }
      };

      // Add event listener to the textbox for "keydown" event
      textbox.addEventListener("keydown", checkForEnterKey);

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
      var start_time = performance.now();
    }
  }
  HtmlFreeRecallPlugin.info = info;

  return HtmlFreeRecallPlugin;
})(jsPsychModule);
