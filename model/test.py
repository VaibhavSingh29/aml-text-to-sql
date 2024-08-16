# https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
from transformers.models.auto import AutoModelForSeq2SeqLM
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from utils.beam import RABeamScorer
from transformers import BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor
import torch

path = "experiments/t5conditional/checkpoint-2000"
tokenizer = T5TokenizerFast.from_pretrained(path)
model = T5ForConditionalGeneration.from_pretrained(path)

num_beams=8

model.eval()

encoder_input_str = [
    "How many singers do we have? | stadium : stadium_id, location, name, capacity, highest, lowest, average | singer : singer_id, name, country, song_name, song_release_year, age, is_male | concert : concert_id, concert_name, theme, stadium_id, year | singer_in_concert : concert_id, singer_id",
    "What is the average, minimum, and maximum age of all singers from France? | stadium : stadium_id, location, name, capacity, highest, lowest, average | singer : singer_id, name, country, song_name, song_release_year, age, is_male | concert : concert_id, concert_name, theme, stadium_id, year | singer_in_concert : concert_id, singer_id",
    "What are the different countries with singers above age 20? | stadium : stadium_id, location, name, capacity, highest, lowest, average | singer : singer_id, name, country, song_name, song_release_year, age, is_male | concert : concert_id, concert_name, theme, stadium_id, year | singer_in_concert : concert_id, singer_id",
    "What are all the song names by singers who are older than average? | stadium : stadium_id, location, name, capacity, highest, lowest, average | singer : singer_id, name, country, song_name, song_release_year, age, is_male | concert : concert_id, concert_name, theme, stadium_id, year | singer_in_concert : concert_id, singer_id",
    "What is the average and maximum capacities for all stadiums? | stadium : stadium_id, location, name, capacity, highest, lowest, average | singer : singer_id, name, country, song_name, song_release_year, age, is_male | concert : concert_id, concert_name, theme, stadium_id, year | singer_in_concert : concert_id, singer_id",
    "What is the name and capacity for the stadium with highest average attendance? | stadium : stadium_id, location, name, capacity, highest, lowest, average | singer : singer_id, name, country, song_name, song_release_year, age, is_male | concert : concert_id, concert_name, theme, stadium_id, year | singer_in_concert : concert_id, singer_id",
    "What is the name and capacity of the stadium with the most concerts after 2013? | stadium : stadium_id, location, name, capacity, highest, lowest, average | singer : singer_id, name, country, song_name, song_release_year, age, is_male | concert : concert_id, concert_name, theme, stadium_id, year | singer_in_concert : concert_id, singer_id",
    "What are the names of the stadiums without any concerts? | stadium : stadium_id, location, name, capacity, highest, lowest, average | singer : singer_id, name, country, song_name, song_release_year, age, is_male | concert : concert_id, concert_name, theme, stadium_id, year | singer_in_concert : concert_id, singer_id"
]

encoder_input_ids = tokenizer(encoder_input_str[6], padding=True, truncation=True, return_tensors='pt').input_ids

print(model.config.decoder_start_token_id)

input_ids = torch.ones((8, 1), device=model.device, dtype=torch.long)
input_ids = input_ids * model.config.decoder_start_token_id

beam_scorer = BeamSearchScorer(
    batch_size=1,
    num_beams=8,
    device=model.device,
    length_penalty=0.1,
)

# beam_scorer= RABeamScorer(
#     batch_size=8, 
#     num_beams = 8, 
#     device='cpu', 
#     tokenizer=tokenizer
# )

logits_processor = LogitsProcessorList(
    [
        MinLengthLogitsProcessor(16, eos_token_id=model.config.eos_token_id),
    ]
)


# add encoder_outputs to model keyword arguments
model_kwargs = {
    "encoder_outputs": model.get_encoder()(
        encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
    )
}


outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

