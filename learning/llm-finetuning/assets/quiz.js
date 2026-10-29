// Reusable quiz widget. Markup per question:
//
//   <div class="quiz-q" data-answer="2">
//     <p class="q">Question text</p>
//     <ol class="opts"><li>A</li><li>B</li><li>C</li><li>D</li></ol>
//     <p class="why">Explanation shown after answering.</p>
//   </div>
//
// data-answer is the 0-based index of the correct option. Wrap the questions
// in <section class="quiz"> and include quiz.css rules via course.css + below.

(function () {
  const style = document.createElement("style");
  style.textContent = `
    .quiz-q { margin: 1.6rem 0; }
    .quiz-q .q { font-weight: 600; margin-bottom: 0.5rem; }
    .quiz-q ol.opts { list-style: none; padding: 0; margin: 0; display: grid; gap: 0.4rem; }
    .quiz-q button { width: 100%; text-align: left; font: inherit; font-size: 0.92rem; color: var(--fg);
      background: var(--bg); border: 1px solid var(--rule); border-radius: 6px; padding: 0.55rem 0.8rem; cursor: pointer; }
    .quiz-q button:hover:not(:disabled) { border-color: var(--accent); }
    .quiz-q button:disabled { cursor: default; }
    .quiz-q button.right { background: var(--good-bg); border-color: var(--good); color: var(--good); }
    .quiz-q button.wrong { background: var(--bad-bg); border-color: var(--bad); color: var(--bad); }
    .quiz-q .why { display: none; font-size: 0.9rem; margin-top: 0.6rem; padding-left: 0.8rem; border-left: 2px solid var(--rule); }
    .quiz-q.done .why { display: block; }
    .quiz-score { font-weight: 600; margin-top: 1rem; }
  `;
  document.head.appendChild(style);

  document.querySelectorAll("section.quiz").forEach((quiz) => {
    const questions = quiz.querySelectorAll(".quiz-q");
    let answered = 0, correct = 0;
    const score = document.createElement("p");
    score.className = "quiz-score";
    quiz.appendChild(score);

    questions.forEach((q, qi) => {
      const answer = Number(q.dataset.answer);
      const items = q.querySelectorAll("ol.opts > li");
      const letters = "ABCD";
      items.forEach((li, i) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = `${letters[i]}. ${li.textContent}`;
        li.textContent = "";
        li.appendChild(btn);
        btn.addEventListener("click", () => {
          const buttons = q.querySelectorAll("button");
          buttons.forEach((b) => (b.disabled = true));
          buttons[answer].classList.add("right");
          if (i !== answer) btn.classList.add("wrong");
          else correct++;
          answered++;
          q.classList.add("done");
          score.textContent = answered === questions.length
            ? `Score: ${correct} / ${questions.length}. Retry the misses tomorrow, from memory.`
            : "";
        });
      });
    });
  });
})();
