const startButton = document.getElementById('start-btn');
const nextButton = document.getElementById('next-btn');
const questionContainerElement = document.getElementById('question-container');

const questionElement = document.getElementById('question');
const answerButtonsElement = document.getElementById('answer-buttons');
const answer_arr = [];

let shuffledQuestions, currentQuestionIndex

startButton.addEventListener('click', startGame);
nextButton.addEventListener('click', ()=>{
    currentQuestionIndex++;
    setNextQuestion();
});

function startGame() {
    startButton.classList.add('hide');
    shuffledQuestions = questions
    // .sort(()=>Math.random() - .5);
    currentQuestionIndex = 0;
    questionContainerElement.classList.remove('hide')
    setNextQuestion();
}

function setNextQuestion() {
    resetState();
    showQuestion(shuffledQuestions[currentQuestionIndex])
}

function showQuestion(question){
    questionElement.innerText = question.question
    question.answers.forEach(answer => {
        const button = document.createElement('button');
        button.innerText = answer.text
        button.classList.add('btn');
        // if(answer.correct) {
        //     button.dataset.correct = answer.correct;
        // }
        button.addEventListener('click', selectAnswer);
        console.log(answer_arr);
        answerButtonsElement.appendChild(button)
    })
}

function resetState() {
    clearStatusClass(document.body)
    nextButton.classList.add('hide');
    while(answerButtonsElement.firstChild) {
        answerButtonsElement.removeChild(answerButtonsElement.firstChild)
    }
}
function selectAnswer(e) {
    const selectedButton = e.target
    if(answer_arr.length < currentQuestionIndex + 1){
        answer_arr.push(selectedButton.innerText);
        selectedButton.classList.add('render');
    } else{
        answer_arr[currentQuestionIndex] = selectedButton
        Array.from(answerButtonsElement.children).forEach(button => {
            button.classList.remove('render');
        });
        selectedButton.classList.add('render');
    }
    // const correct = selectedButton.dataset.correct
    // setStatusClass(document.body, correct)
    // Array.from(answerButtonsElement.children).forEach(button => {
    //     setStatusClass(button, button.dataset.correct)
    // })
    if(shuffledQuestions.length > currentQuestionIndex + 1){
        nextButton.classList.remove('hide');
    } else{
        //Here a link needs to be provided or etc.
        startButton.innerText = 'Click here to view assets you should invest in!!!'
        console.log(answer_arr)
        let amount = 0;
        for(let i = 0; i < answer_arr.length; i++){
            if(answer_arr[i].length < 5){
                amount = answer_arr[i].slice(1);
            }
        }
        //scale from -10 to 10, -10 being risk loving
        const aversion_index = aversion(amount)
        startButton.classList.remove('hide')
    }
}

function aversion(amount){
    const lottery_prize = 200
    const max_willingness = amount
    const prob = 0.5
    const income = 0

    const z = lottery_prize
    const lamb = max_willingness  // can't use lambda because it's python keyword
    const alpha = prob
    // calculate rel_risk_aversion
    const rel_risk_aversion = (alpha * z - lamb) / (lamb**2 / 2 +
                                            alpha * z**2 / 2 - alpha * lamb * z)
    const pho = rel_risk_aversion
    // gradient of relative risk aversion in terms of max willingness to pay
    const grad_pho_over_lamb = -(pho/(alpha*z-lamb)+pho**2)

    // relative risk premium
    const c = 0.00125 * pho * income
    return rel_risk_aversion * 1000;
}

function setStatusClass(element) {
    clearStatusClass(element)
    element.classList.add('render')
    // if(correct){
    //     element.classList.add('correct')
    // } else{
    //     element.classList.add('wrong')
    // }
}

function clearStatusClass(element) {
    element.classList.remove('render')
    // element.classList.remove('wrong')
}
const questions = [
    {
        question: 'What is your gender?',
        answers: [
            { text: 'Male'},
            { text: 'Female'}
        ]
    }, 
    {
        question: 'What is your age?',
        answers: [
            { text: '10'},
            { text: '20'}
        ]
    },     {
        question: 'What is your highest level of education?',
        answers: [
            { text: 'Middle school and below'},
            { text: 'High School'},
            {text: 'College and above'}
        ]
    },     {
        question: 'What is your yearly income?',
        answers: [
            { text: '$0- $60,000'},
            { text: '$60,000 - $150,000'},
            { text: '$150,000 and above'}
        ]
    },      {
        question: 'What is your employment status?',
        answers: [
            { text: 'unemployed'},
            { text: 'self employed'},
            { text: '$150,000 and above'}
        ]
    },      {
        question: 'Do you have any health impairment/disabilitys?',
        answers: [
            { text: 'Yes'},
            { text: 'No'},
        ]
    },      {
        question: 'Are you married?',
        answers: [
            { text: 'Yes'},
            { text: 'No'},
        ]
    },{
        question: 'Now some fun questions! This question is very important to help you find assets that you should invest in. Imagine that you earn 200 dollars a month. What would you prefer?',
        answers: [
            {text: 'Click here and then click next to continue'}
        ]
    }, {
        question: 'Which do you prefer?',
        answers: [
            {text: '$50'},
            {text: 'A fair coin flip in which you get $200 if it is heads, $0 if it is tails'}
        ]
    }, {
        question: 'Which do you prefer?',
        answers: [
            {text: '$60'},
            {text: 'A fair coin flip in which you get $200 if it is heads, $0 if it is tails'}
        ]
    }, {
        question: 'Which do you prefer?',
        answers: [
            {text: '$70'},
            {text: 'A fair coin flip in which you get $200 if it is heads, $0 if it is tails'}
        ]
    }, {
        question: 'Which do you prefer?',
        answers: [
            {text: '$80'},
            {text: 'A fair coin flip in which you get $200 if it is heads, $0 if it is tails'}
        ]
    }, {
        question: 'Which do you prefer?',
        answers: [
            {text: '$90'},
            {text: 'A fair coin flip in which you get $200 if it is heads, $0 if it is tails'}
        ]
    }, {
        question: 'Which do you prefer?',
        answers: [
            {text: '$100'},
            {text: 'A fair coin flip in which you get $200 if it is heads, $0 if it is tails'}
        ]
    }, 
    {
        question: 'Which do you prefer?',
        answers: [
            {text: '$110'},
            {text: 'A fair coin flip in which you get $200 if it is heads, $0 if it is tails'}
        ]
    }, {
        question: 'Which do you prefer?',
        answers: [
            {text: '$120'},
            {text: 'A fair coin flip in which you get $200 if it is heads, $0 if it is tails'}
        ]
    }
];