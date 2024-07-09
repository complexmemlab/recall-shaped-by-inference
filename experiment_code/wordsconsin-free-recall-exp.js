const jsPsych = initJsPsych({
  on_start: () => {
    let subid = jsPsych.data.getURLVariable("PROLIFIC_PID");
    jsPsych.data.addProperties({ subid: subid });
  },
  on_finish: () => {
    jsPsych.data.addProperties({ experiment: "word-search-task-FR" });
  },
});

const nr_trials = 224;
const nr_runs = 4;
const runs = [
  { run: 1, rules: [1, 2, 3, 4], sizes: [6, 7, 8] },
  { run: 2, rules: [1, 2, 3, 4], sizes: [6, 7, 8] },
  { run: 3, rules: [1, 2, 3, 4], sizes: [6, 7, 8] },
  { run: 4, rules: [1, 2, 3, 4], sizes: [6, 7, 8] },
];
let listOfStims = [smallNatural, smallManmade, bigNatural, bigManmade];
let targets = [];
const ruleTypes = ["small", "large", "natural", "manmade"];

for (let i = 0; i < listOfStims.length; i++) {
  listOfStims[i] = jsPsych.randomization.shuffle(listOfStims[i]);
  targets.push(listOfStims[i]);
}
// Initialize 'targets' for each run
runs.forEach((run) => {
  run.targets = [];
});

// Divide each array in 'targets' into fourths and distribute to runs
targets.forEach((targetArray) => {
  const fourthLength = Math.floor(targetArray.length / 4);
  runs.forEach((run, index) => {
    const startIdx = index * fourthLength;
    const endIdx =
      index === runs.length - 1 ? undefined : (index + 1) * fourthLength; // Take till the end for the last segment
    run.targets.push(targetArray.slice(startIdx, endIdx));
  });
});

// turn blocks into objects
const genBlockObjList = () => {
  let conditions = [1, 2, 3, 4];
  let blockLens = [6,6,7,7,7,7,8, 8];
  let blockObjList = [];
  for (let i = 0; i < conditions.length; i++) {
    for (let j = 0; j < blockLens.length; j++) {
      let blockObj = {
        rule: conditions[i],
        rule_word: ruleTypes[i],
        blockLen: blockLens[j],
      };
      blockObjList.push(blockObj);
    }
  }
  return blockObjList;
}
// shuffle the blockObjList
const shuffleBlockObjList = (currBlockObjList) => {
  var foundCond = false;
  while (!foundCond) {
    for (let i = 1; i < currBlockObjList.length; i++) {
      let currItem = currBlockObjList[i].rule;
      let prevItem = currBlockObjList[i - 1].rule;
      if (currItem === prevItem) {
        currBlockObjList = jsPsych.randomization.shuffle(currBlockObjList);
        break;
      } else if (i === currBlockObjList.length - 1) {
        foundCond = true;
      }
    }
  }
  let shuffledBlockObjList = currBlockObjList;
  return shuffledBlockObjList;
}

const fillBlockObjList = (blockObjList, runTargets) => {
  let ruleTrials = [];
  for (let i = 0; i < blockObjList.length; i++) {
    ruleTrials.push(
      Array(blockObjList[i].blockLen).fill(blockObjList[i].rule_word),
    );
  }
  ruleTrials = ruleTrials.flat();
  const learnedMat = {
    1: {
      2: [0, 1, 2, 3],
      3: [1, 2],
      4: [0, 3],
    },
    2: {
      1: [0, 1, 2, 3],
      3: [0, 3],
      4: [1, 2],
    },
    3: {
      1: [1, 2],
      2: [0, 3],
      4: [0, 1, 2, 3],
    },
    4: {
      1: [0, 3],
      2: [1, 2],
      3: [0, 1, 2, 3],
    },
  };
  // this is the matrix that will be used to determine which word to choose for the first item

  const discrimMap = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3],
    3: [1, 2],
  };

  const nonDiscrimMap = {
    0: [0, 3],
    1: [1, 2],
    2: [1, 2],
    3: [0, 3],
  };
  // setting first two items in each list
  let randNum = Math.floor(Math.random() * 4);
  for (let i = 0; i < blockObjList.length; i++) {
    if (i === 0) {
      previousRule = randNum + 1;
    } else {
      previousRule = blockObjList[i - 1].rule;
    }
    currentRule = blockObjList[i].rule;
    while (!learnedMat[previousRule][currentRule] && i === 0) {
      previousRule = Math.floor(Math.random() * 4) + 1;
    }
    while (previousRule == currentRule) {
      previousRule = Math.floor(Math.random() * 4) + 1;
    }
    currentArrLen = learnedMat[previousRule][currentRule].length;
    firstRandIdx = Math.floor(Math.random() * currentArrLen); // needs to be changed to work off of learnedMat array length
    discrim = Math.random() < 0.5 ? 0 : 1;
    secondRandIdx = Math.random() < 0.5 ? 0 : 1;
    blockObjList[i].words = [
      runTargets[learnedMat[previousRule][currentRule][firstRandIdx]].pop(),
    ]; // set first word based on the learnedMat
  }

  let remainingTargets = runTargets.flat();
  remainingTargets = jsPsych.randomization.shuffle(remainingTargets);
  for (let i = 0; i < blockObjList.length; i++) {
    for (let j = 1; j < blockObjList[i].blockLen; j++) {
      blockObjList[i].words = blockObjList[i].words.concat(
        remainingTargets.pop(),
      );
    }
  }

  // now I need to make the full word list by pulling out all the words in order from the blockObjList
  let fullTargList = [];
  for (let i = 0; i < blockObjList.length; i++) {
    for (let j = 0; j < blockObjList[i].words.length; j++) {
      fullTargList.push(blockObjList[i].words[j]);
    }
  }
  return { fullTargList, ruleTrials };
}


// instead I will genBlockObjList once and then use it to fill the blockObjList for each run (will write a function for that, and then the remaining functions should work)
const blockObjList = genBlockObjList();
let fullShuffledBlockObjList = shuffleBlockObjList(blockObjList.slice());

// now I need to write a function that getsRunBlockObjList. This will take in the full list and then pop two elements from each ruleType from a random size (6,7,8) and then return the list that those items were popped into
// the function should modify blockObjList in place because I don't want the other runs to have the same blockObjList items
// I also need to make sure to pull 2 blockLen 6 and 2 blockLen 8 items for each run (randomly sampled across the rules)


const getRunBlockObjList = (blockObjList) => {
  let runBlockObjList = [];
  // get the number of rules in the runBlockObjList 
  let ruleCount = {1:0,2:0,3:0,4:0};
  let blockLenCounts = {6:0,8:0};
  ruleCond = true;
  while (runBlockObjList.length < 2) {
    for (let i = 0; i < blockObjList.length; i++) {
     if (ruleCount[blockObjList[i].rule] < 2 && blockLenCounts[blockObjList[i].blockLen] < 2) {
       runBlockObjList.push(blockObjList[i]);
       ruleCount[blockObjList[i].rule] += 1;
       blockLenCounts[blockObjList[i].blockLen] += 1;
       blockObjList.splice(i, 1); 
       i--; // decrement i because I just removed an item from the list
     }
    }
  }

  while (ruleCond) {
    for (let i = 0; i < blockObjList.length; i++) {
      if (ruleCount[blockObjList[i].rule] < 2 && blockObjList[i].blockLen != 6 && blockObjList[i].blockLen != 8) {
        runBlockObjList.push(blockObjList[i]);
        ruleCount[blockObjList[i].rule] += 1;
        blockObjList.splice(i, 1);
        i--; // decrement i because I just removed an item from the list
      }
    }
    if (Object.values(ruleCount).every((val) => val === 2)) {
      ruleCond = false;
    }
  }
  return runBlockObjList;
}

runs.forEach((run) => {
  run.blockObjList = getRunBlockObjList(fullShuffledBlockObjList);
  run.blockObjList = run.blockObjList.flat();
  run.shuffledBlockObjList = shuffleBlockObjList(run.blockObjList);
  const result = fillBlockObjList(run.shuffledBlockObjList, run.targets);
  run.fullTargList = result.fullTargList;
  run.ruleTrials = result.ruleTrials;
});

// rule value is a string of 2 characters, each character is either 0 or 1
// this corresponds to a first or second option on 2 questions, in order:
// 1. is the word smaller (0 in 0th position) or larger (1 in 0th position) than a backpack?
// 2. is the word a natural (0 in 1st position) or manmade (1 in 1st position) item?

// this should be moved after the block setup so that I can actually pull items from the different target parts
//targets = jsPsych.randomization.shuffle(targets.flat())
fullTargList = [
  runs[0].fullTargList,
  runs[1].fullTargList,
  runs[2].fullTargList,
  runs[3].fullTargList
];
ruleTrials = [runs[0].ruleTrials, runs[1].ruleTrials, runs[2].ruleTrials, runs[3].ruleTrials];

for (let i = 0; i < fullTargList.length; i++) {
  fullTargList[i].type = "target";
  fullTargList[i].position = i;
}

// setting up targets
// rule value is a string of 2 characters, each character is either 0 or 1
// this corresponds to a first or second option on 2 questions, in order:
// 1. is the word smaller (0 in 0th position) or larger (1 in 0th position) than a backpack?
// 2. is the word a natural (0 in 1st position) or manmade (1 in 1st position) item?

// var rule = rules[Math.floor(Math.random() * rules.length)]; // random rule for now, need to also think about how to encode the swapoffs
// var one_trial = [stimulus[0],stimulus[1]]; // random words
var repCount = 0;

const instructions1 = {
  type: jsPsychInstructions,
  pages: [
    "Welcome to this experiment.<br>" +
      "Please read all instructions carefully, otherwise you will not know what to do.",
    "In this experiment, you will see words describing <b>items</b> appear on the screen.<br>" +
      "For each item, you have to decide whether or not it matches a particular rule.",
    'If the item matches the rule, you should press the <b>"F"</b> key on the keyboard.<br>' +
      'If the item does not match the rule, you should press the <b>"J"</b> key on the keyboard.',
    "If you respond correctly, you will receive 1 point, otherwise you will receive 0 points.",
    "Before you start the main task, let’s first do some practice trials.<br>" +
      "For the following five trials, you will have to judge <em>whether the item can swim</em>.<br>" +
      'So, you will have to press "F" if the item on the screen can swim, or "J" if it cannot.<br>' +
      'Press "J" or click on next to start.',
  ],
  show_clickable_nav: true,
  key_forward: "j",
  key_backward: "f",
};
let practiceReps = 0;
let practiceWords = [
  { word: "TURTLE" },
  { word: "IPHONE" },
  { word: "PIRANHA" },
  { word: "DISHWASHER" },
];
const practice1 = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: () => {
    return (
      '<div style="font-size: 60px;">' +
      practiceWords[practiceReps].word +
      "</div>"
    );
  },
  choices: ["f", "j"],
};
const practiceFeedback = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: () => {
    let data = jsPsych.data.get().last(1).values()[0];
    if ((practiceReps === 0 || practiceReps === 2) && data.response === "f") {
      return '<div style="font-size: 60px;">1</div>';
    } else if (
      (practiceReps === 1 || practiceReps === 3) &&
      data.response === "j"
    ) {
      return '<div style="font-size: 60px;">1</div>';
    } else {
      return '<div style="font-size: 60px;">0</div>';
    }
  },
  trial_duration: 1000,
};

const fixation = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: '<div style="font-size: 60px;">+</div>',
  choices: "NO_KEYS",
  response_ends_trial: false,
  trial_duration: 1000,
};

const practiceBlock = {
  timeline: [practice1, practiceFeedback, fixation],
  loop_function: () => {
    practiceReps++;
    if (practiceReps === 4) {
      return false;
    } else {
      return true;
    }
  },
};

const instructions2 = {
  type: jsPsychInstructions,
  pages: [
    "You are almost ready to start with the main experiment!<br>" +
      "From now on, there will be four possible rules.<br><br>" +
      "Two of these rules will be about the <b>size</b> of the item.<br> The other two rules will be about the <b>origin</b> of the item.",
    "The two rules about the size of the item are as follows:<br><br>" +
      "1. Is the thing SMALLER than a backpack?<br>" +
      "2. Is the thing LARGER than a backpack? <br><br>" +
      'So, if the thing on screen is DISHWASHER and you have to judge it by the first rule, you would say NO ("J"). But if you would have to judge it by the second rule, you would say YES ("F").',
    "The two rules about the origin of the thing are as follows:<br><br>" +
      "1. Is the thing NATURAL?<br>" +
      "2. Is the thing MANMADE?<br><br>" +
      'So, if the thing on screen is PIRANHA and you have to judge it by the first rule, you would say YES ("F"). But if you would have to judge it by the second rule, you would say NO ("J").<br>' +
      "You will have 5 seconds to respond on each trial, and will then get a point if you were right.<br>",
    "Here is the trick: at each point in time only one rule is in play, but we are not going to tell you which one it is.<br><br>" +
      "When a rule is in play, it will stay in play for a while before changing to another rule.<br><br>" +
      "This means that you will have to use your responses and the points to figure out the rule.",
    "When the rule in play changes, it will do so without warning.<br>" +
      "To be precise: when a rule is in play, it will stay in play for a while, but eventually it will change to one of the other three rules.<br><br>",
    "In this experiment, rules will follow each other in random order.<br>" +
      "So, if you notice that the rule has switched, then you will have to figure out what the new rule is.",
    "Good luck! As a reminder, here are the four rules:<br><br>" +
      "1. Is the thing SMALLER than a backpack?<br>" +
      "2. Is the thing LARGER than a backpack? <br>" +
      "3. Is the thing NATURAL?<br>" +
      "4. Is the thing MANMADE?<br><br>" +
      "Please remember them, because we will not show you them again.<br>",
    "During this experiment we will ask you to alternate between this task and typing out all the words you can remember from it<br><br>" +
      `After you have done ${nr_trials/nr_runs} trials you will be given instructions on the typing portion.`,
    'If you are ready for the rule guessing task, press "J" or click "next" to start the experiment.',
  ],
  key_forward: "j",
  key_backward: "f",
  show_clickable_nav: true,
};

const pointsForWords = (rep_count, response) => {
  if (ruleTrials[runNumber][repCount] === "small") {
    if (
      ((response === "f") &
        (fullTargList[runNumber][rep_count].rule === "00")) |
      ((response === "f") & (fullTargList[runNumber][rep_count].rule === "01"))
    ) {
      return 1;
    } else if (
      ((response === "j") &
        (fullTargList[runNumber][rep_count].rule === "10")) |
      ((response === "j") & (fullTargList[runNumber][rep_count].rule === "11"))
    ) {
      return 1;
    } else {
      return 0;
    }
  } else if (ruleTrials[runNumber][rep_count] === "large") {
    if (
      ((response === "f") &
        (fullTargList[runNumber][rep_count].rule === "10")) |
      ((response === "f") & (fullTargList[runNumber][rep_count].rule === "11"))
    ) {
      return 1;
    } else if (
      ((response === "j") &
        (fullTargList[runNumber][rep_count].rule === "00")) |
      ((response === "j") & (fullTargList[runNumber][rep_count].rule === "01"))
    ) {
      return 1;
    } else {
      return 0;
    }
  } else if (ruleTrials[runNumber][rep_count] === "natural") {
    if (
      ((response === "f") &
        (fullTargList[runNumber][rep_count].rule === "00")) |
      ((response === "f") & (fullTargList[runNumber][rep_count].rule === "10"))
    ) {
      return 1;
    } else if (
      ((response === "j") &
        (fullTargList[runNumber][rep_count].rule === "01")) |
      ((response === "j") & (fullTargList[runNumber][rep_count].rule === "11"))
    ) {
      return 1;
    } else {
      return 0;
    }
  } else if (ruleTrials[runNumber][rep_count] === "manmade")
    if (
      ((response === "f") &
        (fullTargList[runNumber][rep_count].rule === "01")) |
      ((response === "f") & (fullTargList[runNumber][rep_count].rule === "11"))
    ) {
      return 1;
    } else if (
      ((response === "j") &
        (fullTargList[runNumber][rep_count].rule === "00")) |
      ((response === "j") & (fullTargList[runNumber][rep_count].rule === "10"))
    ) {
      return 1;
    } else {
      return 0;
    }
};

const dim_trial = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: () => {
    return (
      '<div style="font-size: 60px;">' +
      fullTargList[runNumber][repCount].word +
      "</div>"
    );
  },
  prompt:
    "<div style='display:flex;justify-content:space-between;margin-top:100px'><div style='display:flex;flex-direction:column;margin-right:40px'><div style='font-size:20px;'>Yes</div><div style='font-size:20px;'>F</div></div>" +
    "<div style='display:flex;flex-direction:column;'><div style='font-size:20px;'>No</div><div style='font-size:20px;'>J</div></div></div>",
  choices: ["f", "j"],
  trial_duration: 3000, // remember to fix
  on_finish: () => {
    let data = jsPsych.data.get().last(1).values()[0];
    let wordPoints = pointsForWords(repCount, data.response);
    jsPsych.data.addDataToLastTrial({
      points: wordPoints,
      word: fullTargList[runNumber][repCount].word,
      rule: fullTargList[runNumber][repCount].rule,
      current_rule: ruleTrials[runNumber][repCount],
    });
  },
};

const dim_trial_post = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: () => {
    return (
      '<div style="font-size: 60px;">' +
      fullTargList[runNumber][repCount].word +
      "</div>"
    );
  },
  prompt: () => {
    let data = jsPsych.data.get().last(1).values()[0];
    if (data.response === "j") {
      // f grey
      prompt =
        "<div style='display:flex;justify-content:space-between;margin-top:100px'><div style='display:flex;flex-direction:column;margin-right:40px'><div style='font-size:20px;color:#D3D3D3'>Yes</div><div style='font-size:20px;color:#D3D3D3'>F</div></div>" +
        "<div style='display:flex;flex-direction:column;'><div style='font-size:20px;'>No</div><div style='font-size:20px;'>J</div></div></div>";
    } else if (data.response === "f") {
      // j grey
      prompt =
        "<div style='display:flex;justify-content:space-between;margin-top:100px'><div style='display:flex;flex-direction:column;margin-right:40px'><div style='font-size:20px;'>Yes</div><div style='font-size:20px;'>F</div></div>" +
        "<div style='display:flex;flex-direction:column;'><div style='font-size:20px;color:#D3D3D3'>No</div><div style='font-size:20px;color:#D3D3D3'>J</div></div></div>";
    } else {
      // all grey
      prompt =
        "<div style='display:flex;justify-content:space-between;margin-top:100px'><div style='display:flex;flex-direction:column;margin-right:40px'><div style='font-size:20px;color:#D3D3D3'>Yes</div><div style='font-size:20px;color:#D3D3D3'>F</div></div>" +
        "<div style='display:flex;flex-direction:column;'><div style='font-size:20px;color:#D3D3D3'>No</div><div style='font-size:20px;color:#D3D3D3'>J</div></div></div>";
    }
    return prompt;
  },
  choices: "NO_KEYS",
  trial_duration: () => {
    let data = jsPsych.data.get().last(1).values()[0];
    if (data.rt !== null) {
      return 3000 - data.rt;
    } else {
      return 0;
    }
  },
  response_ends_trial: false,
};

const feedback = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: () => {
    let data = jsPsych.data.get().last(2).values()[0];
    let wordPoints = data.points;
    return '<div style="font-size: 60px;">' + wordPoints + "</div>";
  },
  choices: "NO_KEYS",
  trial_duration: 1000,
};

const full_encoding_trial = {
  timeline: [dim_trial, dim_trial_post, feedback, fixation],
  loop_function() {
    repCount++;
    if (repCount == nr_trials / nr_runs) {
      return false;
    } else {
      return true;
    }
  },
};

// need to make a distractor task where people just do multiplication and addition between every run of recall and typing
distractorTimeLimit = 10000;
const distractorTask = {
  timeline: [
    {
      type: jsPsychHtmlKeyboardResponse,
      stimulus: () => {
        A = Math.floor(Math.random() * 10); // draw a random integer between 0 and 9
        B = Math.floor(Math.random() * 10);; // draw a random integer between 0 and 9
        C = Math.floor(Math.random() * 10);; // draw a random integer between 0 and 9
        D = A+B+C - (Math.floor(Math.random() * 2)); // draw a random integer between 0 and 2 and subtract it from A + B + C
      return `Does ${A} + ${B} + ${C} = ${D}?`;
      },
      choices: ["f", "j"],
      prompt: "Press F for False and J for True",
      trial_duration: () => {
        return distractorTimeLimit - (jsPsych.data.get().select('curr_time').values[0] - start_time)
      }
    },
  ],
  loop_function: () => {
    let curr_time = performance.now();
    jsPsych.data.addProperties({curr_time: performance.now()});
    if (curr_time - start_time > distractorTimeLimit) {
      return false; // Need to make sure it forces the end of the trial too
    } else {
      return true;
    }
  },
};
// need to replace the stimulus function in this with an actually good one

const distractorInstructions = {
  timeline: [
    {
      type: jsPsychInstructions,
      pages: [
        "Congratulations! You have finished the first round of the rule guessing task.<br><br>" +
          "During the next part of the experiment, we will ask you to perform a few true or false judgments about math questions",
        "An equation will appear on your screen and you will have to judge whether it is true by pressing 'J' or false by hitting 'F'.<br><br>" +
          "Here is an example: 2 + 3 + 1 = 6<br><br>" +
          "In this instance the equation is True, so you would press 'J'.",
        "You will have as much time as you want to answer each but try and be as quick and accurate as possible.<br>" +
          "After 10 seconds you will be moved to the next task <br><br>" +
          "Try to answer as many as you can!",
        'If you are ready, press "J" or click "next" to begin!',
      ],
      key_forward: "j",
      key_backward: "f",
      show_clickable_nav: true,
    },
  ],
  conditional_function: () => {
    if (runNumber === 0) {
      return true;
    } else {
      return false;
    }
  },
};

const recallInstructions = {
  timeline: [
    {
      type: jsPsychInstructions,
      pages: [
        "Congratulations! You have finished the first round of the rule guessing and math task.<br><br>" +
          "During the next part of the experiment, we will test your memory for the items that you saw during this run of the rule guessing task.",
        "A text box will appear on the screen. Your job will be to type all of the items that you can remember, in any order, into the text box.<br><br>" +
          "Use the space bar to submit each item. Once you submit an item using the space bar, the word will disappear from the text box and you can type in another item.",
        "You will have 3 minutes to think of as many items as possible.<br>" +
          "After 3 minutes, a button will appear that will allow you to finish the memory test.<br><br>" +
          "If you are still able to remember items that you have seen, you should continue to enter them until you can no longer remember any more items.<br>" +
          "Once you are finished typing in items, hit the button to continue to the next part of the experiment.",
        'If you are ready, press "J" or click "next" to begin!',
      ],
      key_forward: "j",
      key_backward: "f",
      show_clickable_nav: true,
    },
  ],
  conditional_function: () => {
    if (runNumber === 0) {
      return true;
    } else {
      return false;
    }
  },
};

const postEnc = {
  timeline: [
    {
      type: jsPsychHtmlKeyboardResponse,
      stimulus: () => {
        return (
          "Congratulations you finished run number " +
          (runNumber + 1) +
          " of the rule guessing task.<br><br>Now it is time for another round of the math task. Remember, we want you to answer whether a given math expression is true or false. Press any key when ready."
        );
      },
    },
  ],
  conditional_function: () => {
    if (runNumber != 0) {
      start_time = performance.now();
      return true;
    } else {
      start_time = performance.now();
      return false;
    }
  },
};

const postDist = {
  timeline: [
    {
      type: jsPsychHtmlKeyboardResponse,
      stimulus: () => {
        return (
          "Congratulations you finished run number " +
          (runNumber + 1) +
          " of the math task.<br><br>Now it is time for another round of the typing task. Remember we want you to type any words that you remember from the most recent round of the guessing task. Press any key when ready."
        );
      },
    },
  ],
  conditional_function: () => {
    if (runNumber != 0) {
      return true;
    } else {
      return false;
    }
  },
};

const postRec = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: () => {
    if (runNumber != nr_runs - 1) {
      return (
        "Congratulations you finished run number " +
        (runNumber + 1) +
        " of the typing task.<br><br>Now it is time for another round of the rule guessing task." +
        "1. Is the thing SMALLER than a backpack?<br>" +
        "2. Is the thing LARGER than a backpack? <br>" +
        "3. Is the thing NATURAL?<br>" +
        "4. Is the thing MANMADE?<br><br>" +
        "Press any key when ready."
      );
    } else {
      return "Congratulations you finished the final run of the typing task.<br><br>Now there are a few questionnaires and you will be finished. Press any key when ready.";
    }
  },
};

const demographics1 = {
  type: jsPsychSurveyText,
  preamble:
    "<p>Before the experiment ends, please answer these questions." +
    "Your responses are confidential.</p>",
  questions: [
    { prompt: "How old are you?", name: "Age" },
    { prompt: "What is your sex?", name: "Sex" },
    { prompt: "What is your gender?", name: "Gender" },
  ],
};

const questionnaire = {
  type: jsPsychSurveyText,
  questions: [
    { prompt: "Do you have any comments about the study?", name: "comment" },
    {
      prompt: "How difficult did you find the guessing task (the first item)",
      name: "difficulty_guess",
    },
    {
      prompt:
        "How often did you find yourself zoning out during the guessing task?",
      name: "zoneout_guess",
    },
    {
      prompt: "How difficult was it to come up with the right rule?",
      name: "difficulty_strat",
    },
    { prompt: "Were any rules harder to switch to?", name: "difficulty_rule" },
    {
      prompt: "How difficult did you find the memory task (the second item)",
      name: "difficulty_memory",
    },
    {
      prompt:
        "Were there any words that seemed more difficult in the guessing task?",
      name: "difficulty_word_guess",
    },
    // {prompt:"Were there any words that seemed more difficult in the memory task?", name:"difficulty_word_mem"}
  ],
};

const debrief = {
  type: jsPsychExternalHtml,
  url: "debrief.html",
  cont_btn: "start",
};

const freeRecallTrial_noButton = {
  type: htmlFreeRecall,
  stimulus: "Type the words you remember:",
  prompt: "<p>Use space to enter your words.</p>",
  response_ends_trial: true,
  // trial_duration: 3000,
};

const freeRecallTrial_Button = {
  type: htmlFreeRecall,
  stimulus: "Type the words you remember:",
  prompt: "<p>Use space to enter your words.</p>",
  response_ends_trial: true,
  button_string: "Click to end recall task",
  button_delay: 0,
  // trial_duration: 3000,
};
let recallTrials = 0;
let time_limit = 180000; //remember to fix
const freeRecallTrials_noButton = {
  timeline: [freeRecallTrial_noButton],
  on_load: () => {
    if (recallTrials == 0) {
      start_time = performance.now();
    }
    recallTrials += 1;
  },
  loop_function: () => {
    let curr_time = performance.now();
    if (curr_time - start_time > time_limit) {
      return false;
    } else {
      return true;
    }
  },
};

const freeRecallTrials_Button = {
  timeline: [freeRecallTrial_Button],
  loop_function: () => {
    let lastTrialData = jsPsych.data.getLastTrialData();
    if (lastTrialData.trials[0].button !== null) {
      return false;
    } else {
      return true;
    }
  },
};
let runNumber = 0;
const fullBlock = {
  timeline: [
    full_encoding_trial,
    distractorInstructions,
    postEnc,
    distractorTask,
    recallInstructions,
    postDist,
    freeRecallTrials_noButton,
    freeRecallTrials_Button,
    postRec,
  ],
  loop_function() {
    runNumber++;
    // turn the following into elif so for each run I can push a different postEnc and postRec trial type
    if (runNumber == nr_runs) {
      return false;
    } else {
      repCount = 0;
      return true;
    }
  },
};

// const finale = {
//   type: jsPsychHtmlKeyboardResponse,
//   stimulus:
//     "This is the completion code: C134TSD4<br>Please press j after copying it. Thanks for participating!",
//   duration: 30000,
//   choices: "j",
// };

jsPsych.run([
  instructions1,
  practiceBlock,
  instructions2,
  fullBlock,
  demographics1,
  questionnaire,
  debrief,
  // finale,
]);