import PromptSuggestionBtn from "./promptsuggestionbtn"

const PromptSuggestion = ({onPromptClick})=>{
    const prompts = [
        "who is founder of stock?",
        "what is difference between mutual fund and stock?",
        "How can maintain Financial things?",
        "Which is best Mutual fund or stock?",
    ]
    return (
        <div className="prompt-sug">
            {prompts.map((prompt,index)=>
            <PromptSuggestionBtn 
            key={`suggestion-${index}`} text={prompt}
            onClick={ ()=>onPromptClick(prompt)}
            />)}
        </div>
    )
}


export default PromptSuggestion