const form = document.querySelector("form");
const button = document.querySelector("button[type='submit']");

form.addEventListener("submit", function () {

    button.innerHTML = "⏳ Generating...";
    button.disabled = true;

});