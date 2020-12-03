#![cfg_attr(not(feature = "std"), no_std)]

// Frame imports.
use frame_support::{decl_module, decl_storage, decl_event, decl_error, dispatch, ensure, debug};
use frame_support::weights::{DispatchClass, Pays};
use frame_system::{self as system, ensure_signed};
use substrate_fixed::types::U32F32;
use sp_std::convert::TryInto;
use sp_std::{
	prelude::*
};

#[cfg(test)]
mod mock;

#[cfg(test)]
mod tests;

/// Configure the pallet by specifying the parameters and types on which it depends.
pub trait Trait: frame_system::Trait {
	/// Because this pallet emits events, it depends on the runtime's definition of an event.
	type Event: From<Event<Self>> + Into<<Self as frame_system::Trait>::Event>;
}

// The pallet's runtime storage items.
decl_storage! {
	trait Store for Module<T: Trait> as SubtensorModule {
		// Stake Values: Map from account to u32 stake ammount.
		pub Stake get(fn stake): map hasher(blake2_128_concat) T::AccountId => u32;

		// Last Emit Block: Last emission block.
		pub LastEmit get(fn block): map hasher(blake2_128_concat) T::AccountId => T::BlockNumber;

		// Total amount staked.
		TotalStake: u32;


	}
}

// Subtensor events.
decl_event!(
	pub enum Event<T> where AccountId = <T as frame_system::Trait>::AccountId {
		// Sent when a Neuron updates their stake.
		StakeAdded(AccountId, u32),

		// Sent when there is emission from a Neuron.
		Emission(AccountId, u32),
	}
);

// Subtensor Errors.
decl_error! {
	pub enum Error for Module<T: Trait> {
		/// Cannot join as a member because you are already a member
		AlreadyActive,
		/// Cannot perform staking or emission unless the Neuron is already subscribed.
		NotActive,
		// Neuron calling emit has no emission.
		NothingToEmit,
		// Neuron updating weights caused overflow.
		WeightVecNotEqualSize,
		// Neuron setting weights are too large. Cause u32 overlfow.
		WeightSumToLarge,
	}
}

// Subtensor Dispatchable functions.
decl_module! {
	pub struct Module<T: Trait> for enum Call where origin: T::Origin {
		// Errors must be initialized if they are used by the pallet.
		type Error = Error<T>;

		// Events must be initialized if they are used by the pallet.
		fn deposit_event() = default;

		/// Emission. Called by an active Neuron with stake in-order to distribute 
		/// tokens to weighted neurons and to himself. The amount emitted is dependent on
		/// the ammount of stake held at this Neuron and the time since last emission.
		/// neurons are encouraged to calle this function often as to maximize
		/// their inflation in the graph.
		#[weight = (0, DispatchClass::Operational, Pays::No)]
		pub fn emit(origin) -> dispatch::DispatchResult {
			// Check that the extrinsic was signed and get the signer.
			// This function will return an error if the extrinsic is not signed.
			// https://substrate.dev/docs/en/knowledgebase/runtime/origin
			let calling_neuron = ensure_signed(origin)?;
			debug::info!("Emit sent by: {:?}", calling_neuron);

			
			ensure!(Stake::<T>::contains_key(&calling_neuron), Error::<T>::NotActive);
			ensure!(LastEmit::<T>::contains_key(&calling_neuron), Error::<T>::NotActive);

			// Get the last emission block.
			// Get the current block.
			// Get the block reward at this current block.
			// Set the current block as last emit.
			let last_block: T::BlockNumber = LastEmit::<T>::get(&calling_neuron);
			let current_block = system::Module::<T>::block_number();
			let block_reward = Self::block_reward(&current_block);
			LastEmit::<T>::insert(&calling_neuron, current_block);
			debug::info!("Last emit block: {:?}", last_block);
			debug::info!("Current block: {:?}", current_block);
			debug::info!("Block reward: {:?}", block_reward);

			// Get the number of elapsed blocks since last emit.
			// Convert to u32f32.
			let elapsed_blocks = current_block - last_block;
			let elapsed_blocks_u32: usize = TryInto::try_into(elapsed_blocks)
			.ok()
			.expect("blockchain will not exceed 2^32 blocks; qed");
			let elapsed_blocks_u32_f32 = U32F32::from_num(elapsed_blocks_u32);
			debug::info!("elapsed_blocks_u32: {:?}", elapsed_blocks_u32);
			debug::info!("elapsed_blocks_u32_f32: {:?}", elapsed_blocks_u32_f32);

			// Get local and total stake.
			// Convert to u32f32.
			// Calculate stake fraction.
			let total_stake: u32  = TotalStake::get();
			let total_stake_u32_f32 = U32F32::from_num(total_stake);
			let local_stake: u32 = Stake::<T>::get(&calling_neuron);
			let local_stake_u32_f32 = U32F32::from_num(local_stake);
			let stake_fraction_u32_f32 = local_stake_u32_f32 / total_stake_u32_f32;
			debug::info!("total_stake_u32_f32 {:?}", total_stake_u32_f32);
			debug::info!("local_stake_u32_f32 {:?}", local_stake_u32_f32);
			debug::info!("stake_fraction_u32_f32 {:?}", stake_fraction_u32_f32);

			// Calculate total emission at this Neuron based on times since last emit
			// stake fraction and block reward.
			let total_emission_u32_f32 = stake_fraction_u32_f32 * block_reward * elapsed_blocks_u32_f32;
			let total_emission_u32 = total_emission_u32_f32.to_num::<u32>();
			debug::info!("total_emission_u32_f32 {:?} = {:?}*{:?}*{:?}", total_emission_u32_f32, stake_fraction_u32_f32, block_reward, elapsed_blocks_u32_f32);
			ensure!(total_emission_u32_f32 > U32F32::from_num(0), Error::<T>::NothingToEmit);


			Self::deposit_event(RawEvent::Emission(calling_neuron, total_emission_u32));

			// Return.
			Ok(())
		}

		// Staking: Adds stake to the stake account for calling Neuron.
		#[weight = (0, DispatchClass::Operational, Pays::No)]
		fn add_stake(origin, stake_amount: u32) -> dispatch::DispatchResult {
			
			// Check sig.
			let neuron = ensure_signed(origin)?;
			debug::info!("add_stake sent by: {:?}", neuron);
			debug::info!("stake_amount {:?}", stake_amount);

			// Update stake at Neuron.
			// TODO (const): transfer from balance pallet.
			Stake::<T>::insert(&neuron, stake_amount);

			// Update total staked storage iterm.
			let total_stake = TotalStake::get();
			TotalStake::put(total_stake + stake_amount); // TODO (const): check overflow.
			debug::info!("total_stake: {:?}", total_stake + stake_amount);

			// Emit event and finish.
			Self::deposit_event(RawEvent::StakeAdded(neuron, stake_amount));
			Ok(())
		}

		// Subscribes the calling Neuron to the active set.
		#[weight = (0, DispatchClass::Operational, Pays::No)]
		fn subscribe(origin, ip: u128, port: u16, ip_type: u8) -> dispatch::DispatchResult {
			
			// Check sig.
			let new_neuron = ensure_signed(origin)?;
			debug::info!("new_neuron sent by: {:?}", new_neuron);

			Ok(())
		}
		
		// Removes Neuron from active set.
		#[weight = (0, DispatchClass::Operational, Pays::No)]
		fn unsubscribe(origin) -> dispatch::DispatchResult {

			// Check sig.
			let old_neuron = ensure_signed(origin)?;
			debug::info!("unsubscribe sent by: {:?}", old_neuron);

			Ok(())
		}
	}
}

impl<T: Trait> Module<T> {

	// Returns the bitcoin block reward from the block step.
	fn block_reward(now: &<T as system::Trait>::BlockNumber) -> U32F32 {
		// Convert block number into u32.
		let elapsed_blocks_u32 = TryInto::try_into(*now)
			.ok()
			.expect("blockchain will not exceed 2^32 blocks; QED.");

		// Convert block number into u32f32 float.
		let elapsed_blocks_u32_f32 = U32F32::from_num(elapsed_blocks_u32);

		// Bitcoin block halving rate was 210,000 blocks at block every 10 minutes.
		// The average substrate block time is 6 seconds.
		// The equivalent halving would be 10 min * 60 sec / 6 sec =  100 * 210,000.
		// So our halving is every 21,000,000 blocks.
		let block_halving = U32F32::from_num(21000000);
		let fractional_halvings = elapsed_blocks_u32_f32 / block_halving;
		let floored_halvings = fractional_halvings.floor().to_num::<u32>();
		debug::info!("block_halving: {:?}", block_halving);
		debug::info!("floored_halvings: {:?}", floored_halvings);

		// Return the bitcoin block reward.
		let block_reward = U32F32::from_num(50);
		// NOTE: Underflow occurs in 21,000,000 * (16 + 4) blocks essentially never.
		let block_reward_shift = block_reward.overflowing_shr(floored_halvings).0;
		block_reward_shift
	}
}